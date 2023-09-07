# -*- coding: utf-8 -*-
"""
Created on Mon Sep 4 12:09:42 2023

@author: mjehl
"""

#Define Battery Parameters

energy_capacity = 100 #@param {type:"number"}
#The maximum volume of energy that can be stored in the battery system, measured in MWh. Note that this is the quantity of energy stored in the system after charging efficiency losses and before discharging efficiency losses.

charge_power_limit = 25 #@param {type:"number"}
#The maximum power rate at which the battery can charge, measured in MW.

discharge_power_limit = 25 #@param {type:"number"}
#The maximum power rate at which the battery can dicharge, measured in MW.

charge_efficiency = 0.95 #@param {type:"number"}
#The efficiency at which energy can enter the battery. For example, charging at 1 MW for 1 hour with a 95% charge efficiency will result in 0.95 MWh of energy stored in the battery.

discharge_efficiency = 0.95 #@param {type:"number"}
#The efficiency at which energy can leave the battery. For example, discharging at 1 MW for 1 hour with a 95% discharge efficiency will result in 0.95 MWh of energy to the grid.

SOC_max = 100 #@param {type:"number"}
#The maximum allowable amount of energy that can be stored in the battery, measured in MWh. There can be longevity or other reasons for making this less than the Energy Capacity.

SOC_min = 0 #@param {type:"number"}
#The minimum allowable amount of energy that can be stored in the battery, measured in MWh. There can be longevity or other reasons for making this greater than zero.

daily_cycle_limit = 1 #@param {type:"number"}
#The maximum number of cycles allowed in a day. This constraint can be imposed for battery health reasons.

annual_cycle_limit = 300 #@param {type:"number"}
#The maximum number of cycles allowed in a year. This constraint can be imposed for battery warranty or long-term degradation limiting reasons.

SOC_initial = 0 #@param {type:"number"}
#The SOC of the first interval of the analysis. For analyses of short periods of time, this can have a meaningful impact on net revenue. For analyses over a month or so, it doesn't matter much.


################################################################################
#Day Ahead Price Retrieval
!pip install pulp
!pip install gridstatusio
!pip install pandas
from pulp import *
import gridstatusio
from gridstatusio import GridStatusClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Get data from gridstatus.io

API_Key = "#####################"
client = GridStatusClient(API_Key)

# Select CAISO Pricing Node
pricing_node = "TH_NP15_GEN-APND" #@param ["TH_ZP26_GEN-APND", "TH_SP15_GEN-APND", "TH_NP15_GEN-APND", "DLAP_VEA-APND", "DLAP_SDGE-APND", "DLAP_SCE-APND", "DLAP_PGE-APND"]

# Select Pricing Date Range (Earliest Data: 2020-01-01)

grid_status_data = client.get_dataset(
    dataset="caiso_lmp_day_ahead_hourly",
    filter_column="location",
    filter_value= pricing_node,
    start = "2022-01-01" #@param {type:"date"}
    ,
    end = "2023-09-01" #@param {type:"date"}
    ,
    tz="US/Pacific",  # return time stamps in Pacific time
)

# Create dataframe for relevant columns and extract prices as a list from it

da_prices_df = grid_status_data[["interval_start_local", "lmp"]]
da_prices = da_prices_df['lmp'].tolist()


####################################################################################
#Dispatch Optimization

# Price Forecast for num_hours hours
num_hours = len(da_prices)
num_days = num_hours / 24
total_cycle_limit = (num_days / 365) * annual_cycle_limit

# Variables
charge_vars = LpVariable.dicts("Charging", range(num_hours), lowBound=0, upBound=charge_power_limit)
discharge_vars = LpVariable.dicts("Discharging", range(num_hours), lowBound=0, upBound=discharge_power_limit)
SOC_vars = LpVariable.dicts("SOC", range(num_hours+1), lowBound=SOC_min, upBound=SOC_max)  # Including initial SOC

# Problem
prob = LpProblem("Battery Scheduling", LpMaximize)

# Objective function
prob += lpSum([da_prices[t]*discharge_efficiency*discharge_vars[t] - da_prices[t]*charge_vars[t]/charge_efficiency for t in range(num_hours)])

# Constraints
# Initial SOC constraint
prob += SOC_vars[0] == SOC_initial

# SOC update constraints
for t in range(num_hours):
    if t == 0:
        prob += SOC_vars[t+1] == SOC_vars[t] + charge_efficiency*charge_vars[t] - discharge_vars[t]
    else:
        prob += SOC_vars[t+1] == SOC_vars[t] + charge_efficiency*charge_vars[t] - discharge_vars[t]

# Charge/Discharge constraints based on SOC
for t in range(num_hours):
    prob += SOC_vars[t] + charge_efficiency*charge_vars[t] <= SOC_max  # Cannot charge if SOC_max is reached
    prob += SOC_vars[t] - discharge_vars[t] >= SOC_min  # Cannot discharge below SOC_min

# Simultaneous charge and discharge constraint
for t in range(num_hours):
    prob += charge_vars[t] + discharge_vars[t] <= max(charge_power_limit, discharge_power_limit)

 # Daily cycle limit constraints
prob += lpSum([charge_vars[t] for t in range(24)]) * charge_efficiency / energy_capacity <= daily_cycle_limit

# Annual cycle limit constraints
prob += lpSum([charge_vars[t] for t in range(num_hours)]) * charge_efficiency / energy_capacity <= total_cycle_limit

# Solve the problem
prob.solve()

# Create Battery Dispatch dataframe with results

discharge_vars_series = pd.Series([value(discharge_vars[t]) for t in range(num_hours)], name='discharge_vars')
charge_vars_series = pd.Series([value(charge_vars[t]) for t in range(num_hours)], name='charge_vars')
soc_vars_series = pd.Series([value(SOC_vars[t]) for t in range(num_hours)], name='soc_vars')

battery_dispatch_df = da_prices_df.assign(discharge_vars=discharge_vars_series, charge_vars=charge_vars_series, SOC_vars=soc_vars_series)

# Set the index to 'interval_start_local'
battery_dispatch_df['interval_start_local'] = pd.to_datetime(battery_dispatch_df['interval_start_local'])
battery_dispatch_df.set_index('interval_start_local', inplace=True)

##########################################################################################
#Parse and Display Results

import pandas as pd

# Calculate hourly metrics
battery_dispatch_df['hourly_discharging_revenue'] = battery_dispatch_df['discharge_vars'] * battery_dispatch_df['lmp'] * discharge_efficiency
battery_dispatch_df['hourly_charging_costs'] = battery_dispatch_df['charge_vars'] * battery_dispatch_df['lmp'] / charge_efficiency
battery_dispatch_df['hourly_net_revenue'] = battery_dispatch_df['hourly_discharging_revenue'] - battery_dispatch_df['hourly_charging_costs']
battery_dispatch_df['hourly_cycles'] = battery_dispatch_df['charge_vars'] * charge_efficiency / energy_capacity

# Aggregate data daily, weekly, monthly, and yearly
cols_to_keep = ['hourly_discharging_revenue', 'hourly_charging_costs', 'hourly_net_revenue', 'hourly_cycles']

daily_metrics = battery_dispatch_df[cols_to_keep].resample('D').sum()
weekly_metrics = battery_dispatch_df[cols_to_keep].resample('W').sum()
weekly_metrics.index = weekly_metrics.index - pd.offsets.Day(6)
monthly_metrics = battery_dispatch_df[cols_to_keep].resample('MS').sum()
yearly_metrics = battery_dispatch_df[cols_to_keep].resample('Y').sum()
yearly_metrics.index = yearly_metrics.index.to_period('Y').to_timestamp('Y')

# Add start and end dates for each period
daily_metrics['End Date'] = (daily_metrics.index + pd.DateOffset(days=1)) - pd.Timedelta(1, unit='s')
weekly_metrics['End Date'] = (weekly_metrics.index + pd.DateOffset(weeks=1)) - pd.Timedelta(1, unit='s')
monthly_metrics['End Date'] = (monthly_metrics.index + pd.offsets.MonthBegin(1)) - pd.Timedelta(1, unit='D')
yearly_metrics['End Date'] = (yearly_metrics.index + pd.DateOffset(years=1)) - pd.Timedelta(1, unit='s')

# Add start and end dates for each period
daily_metrics['Start Date'] = daily_metrics.index
weekly_metrics['Start Date'] = weekly_metrics.index
monthly_metrics['Start Date'] = monthly_metrics.index
yearly_metrics['Start Date'] = yearly_metrics.index

# Determine table metrics based on number of days of analysis
if num_days <= 31:
    # Daily metrics
    metrics = daily_metrics
elif num_days <= 93:
    # Weekly metrics
    metrics = weekly_metrics
elif num_days <= 730:
    # Monthly metrics
    metrics = monthly_metrics
else:
    # Yearly metrics
    metrics = yearly_metrics

# Calculate the total for each column
totals = metrics.sum(numeric_only=True)
totals.name = 'Total'

# Append totals to the end of the dataframe
metrics = pd.concat([metrics, pd.DataFrame(totals).T])

# Rename columns for the final table
metrics = metrics.rename(columns={
    'hourly_discharging_revenue': 'Discharging Revenue ($)',
    'hourly_charging_costs': 'Charging Costs ($)',
    'hourly_net_revenue': 'Net Revenue ($)',
    'hourly_cycles': 'Cycles'
})

# Prepare the values in pandas before passing to Plotly
metrics_no_total = metrics.iloc[:-1].copy() # Exclude the 'Total' row temporarily
metrics_no_total.index = pd.to_datetime(metrics_no_total.index).strftime('%Y-%m-%d %H:%M')
metrics_no_total['End Date'] = metrics_no_total['End Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '')
metrics_no_total['Cycles'] = metrics_no_total['Cycles'].round(1)
metrics_no_total['Discharging Revenue ($)'] = metrics_no_total['Discharging Revenue ($)'].apply(lambda x: f"${x:,.0f}")
metrics_no_total['Charging Costs ($)'] = metrics_no_total['Charging Costs ($)'].apply(lambda x: f"${x:,.0f}")
metrics_no_total['Net Revenue ($)'] = metrics_no_total['Net Revenue ($)'].apply(lambda x: f"${x:,.0f}")

# Handle the 'Total' row separately
total_row = metrics.iloc[-1].copy()
total_row.name = 'Total'
total_row['Start Date'] = ''
total_row['End Date'] = ''
total_row['Cycles'] = f"{total_row['Cycles']:.1f}"
total_row['Discharging Revenue ($)'] = f"${total_row['Discharging Revenue ($)']:,.0f}"
total_row['Charging Costs ($)'] = f"${total_row['Charging Costs ($)']:,.0f}"
total_row['Net Revenue ($)'] = f"${total_row['Net Revenue ($)']:,.0f}"

# Join them back together
metrics = pd.concat([metrics_no_total, total_row.to_frame().T])

# Generate table
table = go.Figure(data=[go.Table(
    header=dict(values=['Start Date', 'End Date', 'Cycles', 'Discharging Revenue ($)', 'Charging Costs ($)', 'Net Revenue ($)'],
                fill_color='black',
                font=dict(color='white'),
                align='left'),
    cells=dict(values=[metrics.index, metrics['End Date'], metrics['Cycles'],
                       metrics['Discharging Revenue ($)'], metrics['Charging Costs ($)'],
                       metrics['Net Revenue ($)']],
               fill_color='darkslategray',
               font=dict(color='white'),
               align='left'))
])

# Find the day with the most net revenue
max_net_revenue_day = daily_metrics['hourly_net_revenue'].idxmax()

# Determine the days before and after
day_before = max_net_revenue_day - pd.Timedelta(days=1)
day_after = max_net_revenue_day + pd.Timedelta(days=1)

# Filter the data for these three days
plot_data = battery_dispatch_df.loc[day_before : day_after]

# Create a subplot with shared x-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add SOC trace
fig.add_trace(
    go.Scatter(x=plot_data.index, y=plot_data['SOC_vars'], name="SOC (MWh)", line=dict(color="blue")),
    secondary_y=False,
)

# Add LMP trace
fig.add_trace(
    go.Scatter(x=plot_data.index, y=plot_data['lmp'], name="LMP ($/MWh)", line=dict(color="red")),
    secondary_y=True,
)

# Set y-axes titles
fig.update_yaxes(title_text="<b>SOC (MWh)</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>LMP ($/MWh)</b>", secondary_y=True)

# Update layout for dark theme
fig.update_layout(
    template="plotly_dark",
    title_text="SOC and LMP for the Day with Maximum Net Revenue and the Adjacent Days",
)


# Render the table and plot
import plotly.io as pio
pio.renderers.default='browser'

table.show()
fig.show()
