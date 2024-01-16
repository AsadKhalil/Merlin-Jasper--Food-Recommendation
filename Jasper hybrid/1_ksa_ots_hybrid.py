#%%

# =============================================================================
# # import Libraries
# =============================================================================

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, date
import argparse
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%

# =============================================================================
# # Setting up Directories
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="The start date of the ots calcuation/prediction.", required=True)
parser.add_argument("--end_date", help="The end date of the ots calcuation/prediction.", required=True)
parser.add_argument("--data_dir", help="Data directory location.", required=True)
args = parser.parse_args()
# data_dir = "/home/munchon/Documents/Munir/Documents/merlin_py/data"

data_base_path = args.data_dir
# data_base_path = data_dir
raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)

#%%

# =============================================================================
# # Loading raw data files from local directory
# =============================================================================

companies_df = pd.read_csv(gap('companies.csv'), usecols=['id' , 'name', 'group_id', 'status', 'is_active_virtual', 'area_id']).rename(columns={'id': 'company_id', 'name': 'company_name', 'status': 'company_status'})
branches_df = pd.read_csv(gap('branches.csv'), usecols=['id' , 'name', 'restaurant_id', 'status']).rename(columns={'id': 'branch_id', 'name': 'branch_name', 'status': 'branch_status'})
companies_schedule_df = pd.read_csv(gap('menu_schedules.csv'), parse_dates=['schedule_on'])
comp_groups_df = pd.read_csv(gap('groups.csv'), usecols=['id', 'status']).rename(columns={'id' : 'group_id', 'status' : 'group_status'})
orders_df = pd.read_csv(gap('orders.csv'), parse_dates=['schedule_on'])
branch_areas_df = pd.read_csv(gap('branch_area.csv'), usecols=['branch_id', 'area_id'])
restaurant_df = pd.read_csv(gap('restaurants.csv'), usecols=['id', 'name', 'status','category_status']).rename(columns={'id': 'restaurant_id'})
restaurant_details_df = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category']).rename(columns={'merlin_3_category': 'restaurant_category', 'merlin_3_type': 'restaurant_type'})

#%%

# =============================================================================
# # Setting up Dates and removing Ramadan period
# =============================================================================

# Fetching Ramadan period
ramzaan_sdate =  '2021-04-12'
ramzaan_edate =  '2021-05-13'
ramzaan_sdate = datetime.strptime(ramzaan_sdate, '%Y-%m-%d')
ramzaan_edate = datetime.strptime(ramzaan_edate, '%Y-%m-%d')
ramzaan_period = [ramzaan_sdate+timedelta(days=x) for x in range((ramzaan_edate-ramzaan_sdate).days)]

# Setting up Scheduling & Window dates
start_date = args.start_date
end_date = args.end_date
# start_date = '2021-06-20'
# end_date = '2021-06-24'

week_start_date =  datetime.fromisoformat(start_date)
week_end_date = datetime.fromisoformat(end_date)

window_start_date = datetime.fromisoformat(start_date)
window_end_date = datetime.fromisoformat(end_date)

window_start_date = window_end_date + relativedelta(weeks=-2)
window_end_date = window_start_date + relativedelta(months=-3)
if (window_end_date < ramzaan_sdate < window_start_date): window_end_date = window_start_date + relativedelta(months=-4)

print("window_start_date :", window_start_date, "window_end_date :", window_end_date)

#%%

# =============================================================================
# # Data Filteration
# =============================================================================

# filter out active companies
companies_df = companies_df.loc[(companies_df['company_status'] == 1) & (companies_df['is_active_virtual'] == 0) & (companies_df['group_id'].notnull())]
companies_df = companies_df.merge(comp_groups_df, on='group_id', how='left')
companies_df = companies_df.loc[companies_df['group_status'] == 1]
companies_df["company_name"] = companies_df["company_name"].str.lower()
companies_df = companies_df[~companies_df["company_name"].str.contains('test')]
companies_df.drop(["group_status"], inplace=True, axis=1)

# filter out active branches
branches_df = branches_df[(branches_df['branch_status'] == 1) | (branches_df['branch_status'] == 2)]
branches_df['branch_name'] = branches_df['branch_name'].str.lower()
branches_df = branches_df[~branches_df["branch_name"].str.contains('test')]

# filter out branches with active restaurants and type/categories
mask = (restaurant_df['status'].isin([1, 2]) & restaurant_df['category_status'].isin([1, 3]))
restaurant_df = restaurant_df.loc[mask]
restaurant_df.drop(["status", "category_status"], inplace=True, axis=1)
restaurant_df["name"] = restaurant_df["name"].str.lower()
mask = ~(restaurant_df["name"].str.contains('test'))
restaurant_df = restaurant_df.loc[mask]
restaurant_df.drop(["name"], inplace=True, axis=1)

# filter out restaurant type and category active ones
restaurant_details_df = restaurant_details_df.loc[(restaurant_details_df['restaurant_type'].notnull()) & (restaurant_details_df['restaurant_category'].notnull())]
restaurant_df = restaurant_df.merge(restaurant_details_df, on='restaurant_id', how='inner')
resto_branches_df = branches_df.merge(restaurant_df, on='restaurant_id', how='inner')

#%%

# logging text

#%%

# =============================================================================
# # Cross data of Companies, Restos & Branches
# =============================================================================

# cross-join companies with branches
def cartesian_product_basic(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))


crossed_df = cartesian_product_basic(companies_df, resto_branches_df)
print("step 1 crossed-data shape is :", crossed_df.shape)

# drop unnecessary columns
crossed_df.drop(columns=['company_status', 'is_active_virtual', 'branch_status'], inplace=True)

# check company and branch area should be same
branch_areas_df = branch_areas_df.groupby('branch_id')['area_id'].apply(list).reset_index(name='branch_areas')
crossed_df = crossed_df.merge(branch_areas_df, on=['branch_id'], how='left')
crossed_df = crossed_df[crossed_df.apply(lambda x: x['area_id'] in x['branch_areas'], axis=1)]

# Optional data validation
print("step_2 crossed-data shape is", crossed_df.shape)

#%%

# =============================================================================
# # Loading Order data
# =============================================================================

# Filter orders data by excluding Ramadan period, Scheduling Start & End date and Orders status is Delivered 
orders_df = orders_df[(~orders_df['schedule_on'].isin(ramzaan_period)) & (orders_df['schedule_on'] <= window_start_date) & (orders_df['schedule_on'] >= window_end_date) & (orders_df['status']==5)]
# Merge orders_df with resto_branches_df
orders_compiled_1 = orders_df.merge(resto_branches_df, on='branch_id', how='left')
# Sum total orders
orders_compiled_1 = orders_compiled_1.groupby(['company_id', 'restaurant_id'])['quantity'].sum().reset_index()
# Merge orders_df with crossed_df to get "total_orders"
crossed_df = crossed_df.merge(orders_compiled_1, on=['company_id', 'restaurant_id'], how="left")
crossed_df.rename(columns = {'quantity': 'total_orders'}, inplace = True)

# =============================================================================
# # Loading Schedules data
# =============================================================================

# Remove schedules where deleted_at is not null
companies_schedule_df = companies_schedule_df[companies_schedule_df['deleted_at'].isna()]
# Filter orders data by excluding Ramadan period, Scheduling Start & End date and Orders status is Delivered 
companies_schedule_df = companies_schedule_df.loc[(~companies_schedule_df['schedule_on'].isin(ramzaan_period)) & (companies_schedule_df['schedule_on'] >= window_end_date) & (companies_schedule_df['schedule_on'] <= window_start_date)]
# Merge schedules dataframe with branches_df to get restaurant_id
companies_schedule_df = pd.merge(companies_schedule_df,branches_df[['branch_id','restaurant_id']], on='branch_id', how='left')
# Remove rows where restaurant_id is null beacause there are inactive branches or restaurants
companies_schedule_df = companies_schedule_df.loc[companies_schedule_df['restaurant_id'].notnull()]
# get schedules count of Company & Resto in last 90 days    
companies_schedule_df = companies_schedule_df.groupby(["company_id", "restaurant_id"])["id"].count().reset_index(name="times_scheduled")
# Merge schdules count dataframe with crossed_df to get "times_scheduled"
crossed_df = crossed_df.merge(companies_schedule_df, on=["company_id", "restaurant_id"], how="left")


#%%

# =============================================================================
# # Restaurants Grouping
# =============================================================================

# Create a dataframe for list of restaurants with total orders in last 90 days
resto_orders = pd.merge(branches_df[['branch_id','restaurant_id']], orders_df[['branch_id','schedule_on','quantity']], on='branch_id', how='inner')
resto_orders = resto_orders.groupby(['restaurant_id'])['quantity'].sum().reset_index()
resto_orders = resto_orders.sort_values('quantity', ascending=False).reset_index(drop=True)
resto_orders.rename(columns = {'quantity': '90days_orders'}, inplace = True)

# Compute Quantiles for Restaurants grouping
Q1 = resto_orders['90days_orders'].quantile(q=.25,interpolation='higher')
Q2 = resto_orders['90days_orders'].quantile(q=.50,interpolation='higher')
Q3 = resto_orders['90days_orders'].quantile(q=.75,interpolation='higher')
Q4 = resto_orders['90days_orders'].max()

print("\n","Q1 =",Q1,",","Q2 =",Q2,",","Q3 =",Q3,",","Q4 =",Q4)

# Tag Restos with their respective Groups 
resto_orders.loc[resto_orders['90days_orders'] <= Q1, "resto_groups"] = "D"
resto_orders.loc[(resto_orders['90days_orders'] > Q1) & (resto_orders['90days_orders'] <= Q2), "resto_groups"] = "C"
resto_orders.loc[(resto_orders['90days_orders'] > Q2) & (resto_orders['90days_orders'] <= Q3), "resto_groups"] = "B"
resto_orders.loc[(resto_orders['90days_orders'] > Q3) & (resto_orders['90days_orders'] <= Q4), "resto_groups"] = "A"

# Restaurants grouping detail summary
restos_summary = resto_orders.groupby(['resto_groups']).agg({'restaurant_id':'count','90days_orders':['min','max']})
restos_summary.columns = restos_summary.columns.droplevel(0)
restos_summary = restos_summary.reset_index()
restos_summary.rename(columns = {'count': 'restos_count','min':'orders_min','max':'orders_max'}, inplace = True)
print("\n",restos_summary)

# write restos_grouping file to local drive
resto_orders.to_csv(gapp('resto_groups.csv'), index=False)

# Attached Restos grouping to Crossed_df
crossed_df = pd.merge(crossed_df, resto_orders[['restaurant_id','resto_groups']], on='restaurant_id', how='left')

#%%

# =============================================================================
# # Companies Grouping
# =============================================================================

# Create dataframe for companies with order history
comp_orders = crossed_df.groupby(['company_id']).size().reset_index(name='counts')
comp_orders.drop(['counts'], inplace=True, axis=1)
comp_orders = pd.merge(comp_orders,orders_df[['company_id', 'branch_id', 'schedule_on', 'quantity']], on='company_id', how='left')
comp_orders_non_zero = comp_orders.loc[comp_orders['branch_id'].notnull()].groupby(['company_id', 'schedule_on'])['quantity'].sum().reset_index()

# Create dataframe for companies having zero orders in last 90 days
# Filter companies with zero orders in last 90 days 
comp_with_zero_orders = comp_orders.loc[comp_orders['branch_id'].isnull()]
comp_with_zero_orders = pd.DataFrame(comp_with_zero_orders['company_id']).reset_index(drop=True)

# List of schedule dates of last 90 days
schedule_dates = comp_orders.loc[comp_orders['schedule_on'].notnull()].groupby(['schedule_on']).size().reset_index(name='count')
schedule_dates.drop(['count'], inplace=True, axis = 1)

# Cartesian product of "companies with zero orders" to "list of schedule dates"
zero_orders_comp_df = cartesian_product_basic(comp_with_zero_orders, schedule_dates)
zero_orders_comp_df['quantity'] = 0

# Merge two dataframes
comp_orders_compiled = pd.concat([comp_orders_non_zero, zero_orders_comp_df], axis=0, sort=False)

# Extract week_no from "Schedule_on" as week starts from Sunday
comp_orders_compiled['week_no'] = (comp_orders_compiled['schedule_on'] + pd.DateOffset(days=1)).dt.week + 1
comp_orders_compiled['week_no'] = "W_" + comp_orders_compiled['week_no'].astype(str)

# Again compile orders dataframe for companies
comp_orders_compiled = comp_orders_compiled.groupby(['company_id','week_no'])['quantity'].sum().reset_index(name='total_orders')
comp_orders_compiled = comp_orders_compiled.groupby(['company_id']).agg({'week_no':'count','total_orders':'sum'})
comp_orders_compiled = comp_orders_compiled.reset_index()
comp_orders_compiled.rename(columns = {'week_no': 'weeks_count','total_orders':'orders_sum'}, inplace = True)
comp_orders_compiled['order_avg'] = round(comp_orders_compiled['orders_sum'] / comp_orders_compiled['weeks_count'],0)
comp_orders_compiled['all_avg'] = round(comp_orders_compiled['orders_sum'] / comp_orders_compiled['weeks_count'].nunique(),0)
comp_orders_compiled[['orders_sum','order_avg','all_avg']] = comp_orders_compiled[['orders_sum','order_avg','all_avg']].applymap(np.int64)

# Grouping of companies to run model based on rules
comp_orders_compiled.loc[(comp_orders_compiled['all_avg'] == 0), 'Model'] = 'OTS'
comp_orders_compiled.loc[(comp_orders_compiled['all_avg'] >= 7) & (comp_orders_compiled['weeks_count'] >= round(comp_orders_compiled.weeks_count.nunique() * 0.70,0)), 'Model'] = 'Jasper'
comp_orders_compiled.loc[(comp_orders_compiled['Model'].isnull()) & (comp_orders_compiled['all_avg'] >= 4) & (comp_orders_compiled['weeks_count'] >= round(comp_orders_compiled.weeks_count.nunique() * 0.50,0)), 'Model'] = 'Jasper'
comp_orders_compiled.loc[(comp_orders_compiled['Model'].isnull()) & (comp_orders_compiled['weeks_count'] >= round(comp_orders_compiled.weeks_count.nunique() * 0.70,0)), 'Model'] = 'Jasper'
comp_orders_compiled.loc[(comp_orders_compiled['Model'].isnull()), 'Model'] = 'OTS'
comp_orders_compiled.Model.value_counts(dropna=False)

# Write a csv file of "comp_groups" to the local repository ** Important **
comp_orders_compiled.to_csv(gapp(str('comp_groups.csv')),index=False)

#%%

# =============================================================================
# # Merge Crossed_df with companies grouping dataframe
# =============================================================================

crossed_df = pd.merge(crossed_df, comp_orders_compiled[['company_id', 'Model']], on='company_id', how='left')

#%%

# =============================================================================
# # OTS Calculation
# =============================================================================

# OTS calculation
crossed_df.loc[((crossed_df['total_orders'] != 0) & (crossed_df['total_orders'].notnull()) & (crossed_df['times_scheduled'].notnull())), 'ots'] = round(crossed_df['total_orders']/crossed_df['times_scheduled'], 2)
crossed_df.loc[((crossed_df['total_orders'].isnull()) & (crossed_df['times_scheduled'].notnull()) & (crossed_df['times_scheduled'] > 2)), 'ots'] = 0
crossed_df.loc[((crossed_df['total_orders'].isnull()) & (crossed_df['times_scheduled'].notnull()) & (crossed_df['times_scheduled'] <= 2)), 'ots'] = np.nan
crossed_df.loc[((crossed_df['total_orders'].isnull()) & (crossed_df['times_scheduled'].isnull())), 'ots'] = np.nan
crossed_df.loc[((crossed_df['total_orders'].notnull()) & (crossed_df['times_scheduled'].isnull())), 'ots'] = np.nan
ots_df = crossed_df.copy()

temp_df = ots_df.groupby(['area_id', 'restaurant_category', 'resto_groups']).agg({'total_orders':'sum','times_scheduled':'sum'})
temp_df = temp_df.reset_index()
temp_df['ots_avg_cat'] = round(temp_df['total_orders'] / temp_df['times_scheduled'], 2)
temp_df.loc[temp_df['ots_avg_cat'].isnull(), 'ots_avg_cat'] = 0
temp_df = temp_df[['area_id', 'restaurant_category', 'resto_groups', 'ots_avg_cat']].copy()

ots_df = ots_df.merge(temp_df, on=['area_id', 'restaurant_category', 'resto_groups'], how="left")
ots_df.loc[ots_df['ots'].isnull(),'ots'] = ots_df['ots_avg_cat']
ots_df.loc[ots_df['ots'].isnull(),'ots'] = 0
# ots_df.loc[(ots_df['ots'] >= 0.25) & (ots_df['ots'] <= 0.5), 'ots'] = 1
ots_df['ots'] = round(ots_df['ots'], 0)

# ots_final = ots_df.loc[ots_df['Model'] == 'OTS']
ots_df = ots_df[['company_id','group_id','branch_id','restaurant_id','ots','Model']].copy()

print("No of days before adding days :", ots_df.shape)

# Adding Days
x = pd.date_range(week_start_date,week_end_date - timedelta(days=0),freq="D").strftime("%Y-%m-%d").to_list()
ots_final = ots_df.copy()
ots_final['schedule_on'] = [x] * len(ots_final)
ots_final = ots_final.explode('schedule_on')
ots_final = ots_final[['company_id','group_id','schedule_on','branch_id','restaurant_id','ots','Model']]
ots_pred_comp = ots_final.loc[ots_final['Model'] == 'OTS']


print("No of days after adding days :", ots_final.shape)

ots_final.to_csv(gapp('ksa_ots.csv'), index=False)
ots_pred_comp.to_csv(gapp('ots_pred_comp.csv'), index=False)


# filename = str("W" + week_start_date.strftime("%U") + " " + week_start_date.strftime("%d-%b") + " to " + week_end_date.strftime("%d-%b"))
# ots_df.to_csv(gapp(str('ksa_ots_' + filename + '.csv')), index=False)


#%%





















