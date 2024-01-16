#%%
import timeit
start = timeit.default_timer()
import pandas as pd
import os
from pathlib import Path
import logging as log
from datetime import datetime, timedelta, date
import argparse
import sys
from tqdm import tqdm
from collections import defaultdict

pd.options.display.width = 0
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
#%%


parser = argparse.ArgumentParser()
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--all_prepare_start_date", help="Actual start date from which order prepare script start. This "
                                                     "will override the --prediction_start_date parameter.")
parser.add_argument("--data_dir", help="Data directory location.", required=True)

args = parser.parse_args()
# args.prediction_start_date = '2021-04-04'
# args.data_dir = "D:\\office_work\\KSA\\ds_work_space_ksa\\data"
# args.data_dir = "D:\\office_work\\KSA\\ds_active_work_space\\data"
args.all_prepare_start_date = None
if args.all_prepare_start_date is not None:
    all_prepare_start_date = args.all_prepare_start_date
    START_DATE = datetime.strptime(all_prepare_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')

else:
    prediction_start_date = args.prediction_start_date
    prediction_start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
    START_DATE = prediction_start_date.strftime('%Y-%m-%d')

data_base_path = args.data_dir


RAW_DATA_FILES_PATH = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)
Path(TARGET_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TARGET_OUTPUT_LOG_DIR = os.path.join(processed_base_data_path, 'output/dropped_data_scheduling_dir')
Path(TARGET_OUTPUT_LOG_DIR).mkdir(parents=True, exist_ok=True)


gap = lambda x: os.path.join(RAW_DATA_FILES_PATH, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)

INPUT_FILE_NAME = 'ksa_ots.csv'
INPUT_FILE = gapp(INPUT_FILE_NAME)
#%%
orders_df = pd.read_csv(INPUT_FILE, parse_dates=['schedule_on'])
orders_df.isna().sum()/orders_df.shape[0]
orders_df = orders_df.dropna()


#%%
menus_df = pd.read_csv(gap('menus.csv'), usecols=['restaurant_id', 'category_id', 'status', 'merlin_status'])


#%%
menus_df = menus_df.query('status == 1 and merlin_status == 1 and category_id == 2')
menus_df = menus_df['restaurant_id'].unique()

## Drop inactive menu restaurants
# dropped_data = orders_df[~orders_df['restaurant_id'].isin(menus_df)][['restaurant_id']].drop_duplicates()
# dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_inactive_menu_restaurants.csv'), index=False)
##

orders_df = orders_df[orders_df['restaurant_id'].isin(menus_df)].copy()
del(menus_df)
#%%
from dateutil.relativedelta import relativedelta
END_REFFERENCE_WINDOW_DATE = (orders_df['schedule_on'].min() + pd.Timedelta(-1, unit='d'))#.date()
START_REFFERENCE_WINDOW_DATE = END_REFFERENCE_WINDOW_DATE + relativedelta(months=-3)


# ### Companies Data prepare
companies_table_attributes = ['id','name','area_id', 'status', 'is_active_virtual', 'group_id']
companies_df = pd.read_csv(gap('companies.csv'), usecols=companies_table_attributes).rename(columns={'id': 'company_id'})

print('Checking companies table size :: ', companies_df.shape)


#%%

mask = ((companies_df['status'] == 1) & (companies_df['is_active_virtual'] == 0))

## Drop inactive menu restaurants
dropped_data = companies_df[~mask]
dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_inactive_companies.csv'), index=False)
##

companies_df = companies_df[mask]
print('Checking companies table size after removing is_virtual_active = 0 and status = 1 :: ', companies_df.shape)


companies_df['name'] = companies_df['name'].str.lower()
mask = (~companies_df["name"].str.contains('test'))

## Drop inactive menu restaurants
dropped_data = companies_df[~mask]
dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_test_companies.csv'), index=False)
##

companies_df = companies_df[mask]
print('Checking companies table size after removing test companies :: ', companies_df.shape)
#%%

companies_df.isna().sum()


#%%
companies_df.drop(["name", "status", "is_active_virtual"], inplace=True, axis=1)
companies_df = companies_df.dropna()
print('Checking companies table size after removing NANs :: ', companies_df.shape)
#%%
groups_df = pd.read_csv(gap('groups.csv'), usecols=['id', 'per_day_restaurants_merlin3']).rename(columns={'id': 'group_id', 'per_day_restaurants_merlin3': 'per_day_restaurants'})
companies_df = companies_df.merge(groups_df, on='group_id', how='left')
#%%
companies_df.head()


# ### Restaurants & Branches

# #### Restaurants
restaurant_df = pd.read_csv(gap('restaurants.csv'), usecols=['id', 'name', 'status','category_status']).rename(columns={'id': 'restaurant_id'})
mask = (restaurant_df['status'].isin([1, 2]) & restaurant_df['category_status'].isin([1, 3]))

## Drop inactive menu restaurants
dropped_data = restaurant_df[~mask]
dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_inactive_restaurants.csv'), index=False)
##

restaurant_df = restaurant_df.loc[mask]

restaurant_df.drop(["status", "category_status"], inplace=True, axis=1)
restaurant_df["name"] = restaurant_df["name"].str.lower()

mask = ~(restaurant_df["name"].str.contains('test'))

## Drop inactive menu restaurants
dropped_data = restaurant_df[~mask]
dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_test_restaurants.csv'), index=False)
##

restaurant_df = restaurant_df[mask]
restaurant_df.drop(["name"], inplace=True, axis=1)

#%%

restaurant_details_df  = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])
restaurant_df = restaurant_df.merge(restaurant_details_df, on='restaurant_id', how='inner')
#%%
# #### Branches

branchs_df = pd.read_csv(gap('branches.csv'), usecols=['id' ,'name', 'status', 'restaurant_id']).rename(columns={'id':'branch_id'})

# Dropping inactive branches
mask = (branchs_df['status'] == 1) | (branchs_df['status'] == 2)

## Drop inactive menu restaurants
# dropped_data = branchs_df[~mask]
# dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_inactive_branches.csv'), index=False)
##

branchs_df = branchs_df[mask]


branchs_df['name'] = branchs_df['name'].str.lower()
mask = ~branchs_df["name"].str.contains('test')

## Drop inactive menu restaurants
# dropped_data = branchs_df[~mask]
# dropped_data.to_csv(gapp(TARGET_OUTPUT_LOG_DIR + '/dropped_test_branches.csv'), index=False)
##

branchs_df = branchs_df[mask]
branchs_df.drop(["name", "status"], inplace=True, axis=1)
print(branchs_df.shape)

branch_details_df = pd.read_csv(gap('branch_details.csv'), usecols=['branch_id', 'blackout_period'])

branchs_df = branchs_df.merge(branch_details_df, on='branch_id', how='inner')


branch_areas_df = pd.read_csv(gap('branch_area.csv'), usecols=['branch_id', 'area_id'])
print('Checking branch_area table size :: ', branch_areas_df.shape)
branch_areas_df = branch_areas_df.groupby('branch_id')['area_id'].apply(list).reset_index(name='areas')


# %%
branchs_df = branchs_df.merge(branch_areas_df, on=['branch_id'], how='left')
print(branchs_df.shape)


# %%

branchs_df = branchs_df.merge(restaurant_df, on=['restaurant_id'], how='inner')
print(branchs_df.shape)

#%%

branchs_df.head()
#%%

branchs_with_areas_df = branchs_df.explode('areas').rename(columns={'areas': 'area_id'})


#%%

branchs_with_areas_df.head()


# ### Create master df
master_df = companies_df.merge(branchs_with_areas_df, on='area_id', how='inner')


#%%

master_df.head()


#%%
menu_schedules_df = pd.read_csv(gap('menu_schedules.csv'), usecols=['company_id', 'schedule_on', 'branch_id', 'deleted_at'], parse_dates=['schedule_on'])
menu_schedules_df = menu_schedules_df[menu_schedules_df['schedule_on'].between(START_REFFERENCE_WINDOW_DATE, END_REFFERENCE_WINDOW_DATE)]
menu_schedules_df = menu_schedules_df[menu_schedules_df['deleted_at'].isna()]
_ = menu_schedules_df.pop('deleted_at')
menu_schedules_df = menu_schedules_df[['company_id', 'schedule_on', 'branch_id']].drop_duplicates(subset=['company_id', 'branch_id'], keep='last')
print('Checking menu_schedules table size :: ', menu_schedules_df.shape)
menu_schedules_df.reset_index(drop=True, inplace=True)
menu_schedules_df.rename(columns={'schedule_on': 'last_schedule_on'}, inplace=True)


#%%

master_df = master_df.merge(menu_schedules_df, on=['company_id', 'branch_id'], how='left')

#%%
# Remove duplicates from the master_df dataframe
print('Before Drop duplicates if any occur in merging', master_df.shape)
master_df = master_df.drop_duplicates(['company_id', 'restaurant_id', 'area_id'])
print('Drop duplicates if any occur in merging', master_df.shape)
#%%

master_df.head()


# ### Attach OTS/Prediction data

#%%

orders_df.head()
#%%
master_df = master_df.merge(orders_df, on=['company_id', 'group_id', 'branch_id', 'restaurant_id'], how='inner')
master_df.rename(columns={'ots':'orders'}, inplace=True)


#%%
master_df.head(10)
#%%

default_date = master_df['schedule_on'].min() + pd.Timedelta(days = -90)
master_df['last_schedule_on'] = master_df['last_schedule_on'].fillna(default_date)
master_df['days_diff'] = master_df['schedule_on'] - master_df['last_schedule_on']
master_df['days_diff'] = master_df['days_diff'].dt.days


#%%
# Create a flag column for "is_not_blackout"
master_df['is_not_blackout'] = 0
master_df.loc[master_df['blackout_period'] <= master_df['days_diff'], 'is_not_blackout'] = 1

master_df['day'] = master_df['schedule_on'].dt.dayofweek


#%%

master_df.isna().sum()
#%%
master_df.shape


# ### Mock section

#%%
master_df.rename(columns={'merlin_3_type': 'type', 'merlin_3_category': 'category'}, inplace=True)
master_df.dropna(inplace=True)

# master_df.drop(columns=['merlin_3_type', 'merlin_3_category'], inplace=True)
# master_df['type'] = -1
# master_df['category'] = -1

# restaurant_types_df = pd.read_csv(gapp('ksa_new_resto_types.csv'))
# restaurant_types_df.head()

# master_df.drop(columns=['type', 'category'], inplace=True)
# master_df = master_df.merge(restaurant_types_df, on=['restaurant_id'], how='inner')
# #%%

# master_df['number_of_companies_in_group'] = master_df.groupby('group_id')['company_id'].transform(lambda x: x.nunique())
# master_df = master_df.query('number_of_companies_in_group > 10')
# master_df.drop(columns=['number_of_companies_in_group'], inplace=True)


#%%
master_df.head()
#%%
master_df = master_df.groupby(['schedule_on', 'branch_id', 'restaurant_id', 'area_id', 'group_id', 'day']).agg(
    per_day_restaurants = ('per_day_restaurants' , max),
    type = ('type', pd.Series.mode),
    category = ('category', pd.Series.mode),
    orders = ('orders', sum),
    is_not_blackout = ('is_not_blackout', min),
    days_diff = ('days_diff', min)
).reset_index()

#%%
# resto_delivery = pd.read_csv(gapp('Delivery Capacity 2 (KSA).csv'), usecols=['Branch ID']).rename(columns={'Branch ID': 'branch_id'})
resto_delivery = pd.read_csv(gap('branches.csv'), usecols=['id', 'delivery_capacity']).rename(columns={'id': 'branch_id'})
# resto_delivery['delivery_capacity'] = 2

master_df = master_df.merge(resto_delivery, on='branch_id', how='left')
master_df['delivery_capacity'] = master_df['delivery_capacity'].fillna(1)

# In[113]:


master_df['per_day_restaurants'] = 3


# In[114]:


master_df.isna().sum()

# %%
# Changing column names and data_types of variables
master_data_numaric_columns_list = list(
    master_df.select_dtypes(include=['float']).columns)
master_df[master_data_numaric_columns_list] = master_df[master_data_numaric_columns_list].astype(
    int)

# master_df.loc[master_df['day'].isin([4, 5]), 'per_day_restaurants'] = 3

master_df = master_df[master_df['per_day_restaurants'] > 2]

# %%
print(master_df.head(2))
mylist = list(master_df.select_dtypes(include=['float']).columns)
master_df[mylist] = master_df[mylist].astype(int)

# %%

master_df['perday_resto'] = master_df['per_day_restaurants']
#%%
# master_df = master_df[master_df['type'].isin([1, 9, 5, 7, 6])]
# master_df = master_df[master_df['category'].isin([1, 2, 8, 14, 17, 18, 21, 22, 23, 27, 28])].copy()

# %%
cusine_table = [
    (23, 'Salads and Healthy Sandwiches', None, None, None),
    (21, 'Mediterranean Sandwiches & wraps', None, None, None),
    (1, 'Arabic & Grills', None, 3, 3),
    (18, 'Italian', 2, 3, 3),
    (17, 'International', 3, 3, 3),
    (27, 'Traditional', 3, 4, 4),
    (2, 'Asian', 2, 3, 3),
    (8, 'Indian', 2, 3, 3),
    (14, 'Burgers & Sandwiches', None, None, None),
    (22, 'Pizza', 2, 3, 3),
    (28, 'Gym Meals', None, None, None),
]
cusine_table = pd.DataFrame(cusine_table, columns=[
                                    'id', 'name', '3', '4', '5']).set_index('id')
cusine_table.drop(columns=['name'], inplace=True)
cusine_table

# %%

scheduled_pairs = pd.read_csv(os.path.join(processed_base_data_path, 'output', 'schedule_pairs.csv'), parse_dates=[
                              'schedule_on']).rename(columns={'schedule_on': 'p_date'})
scheduled_pairs.head()
#%%
# %%
scheduled_pairs = scheduled_pairs.astype({"group_id": int})
join_columns = list(scheduled_pairs.columns)
join_columns.remove('schedule_status')
join_columns

# %%
restaurant_types_df = pd.read_csv(gap('restaurant_types.csv'), usecols=['id', 'name']).rename(
    columns={'id': 'type', 'name': 'restaurant_type_name'})
#%%
# TODO change it to 'left join' after data fix
master_df = master_df.merge(
    restaurant_types_df, on='type', how='left')
#%%
print(master_df.shape)
master_df.rename(columns={'schedule_on': 'p_date'}, inplace=True)
master_df = master_df.merge(scheduled_pairs, on=join_columns, how='left')
print(master_df.shape)
# master_df = master_df[master_df['p_date'] == '2020-02-25']
print(master_df.shape)
master_df.sample(5)

# %%
# Validation Section
master_df['p_date'] = master_df['p_date'].dt.date

# %%

master_df.query('schedule_status == schedule_status').shape

# %%

scheduled_df = master_df.query('schedule_status == 0')
#%%
# %%

all_branches_set = set(master_df['branch_id'].tolist())
scheduled_branches_set = set(scheduled_df['branch_id'].tolist())
set_of_unscaduled_branches = all_branches_set - scheduled_branches_set
# set_of_unscaduled_branches, set_of_companies_without_zones

# %%

Path(gapp(os.path.join('output', 'validation'))).mkdir(
    parents=True, exist_ok=True)

with open(gapp(os.path.join(processed_base_data_path, 'output', 'validation', 'set_of_unscaduled_branches.csv')), 'w') as filehandle:
    for listitem in set_of_unscaduled_branches:
        filehandle.write('%s\n' % listitem)

# %%

branches_rank = master_df.groupby('branch_id')['orders'].sum(
).sort_values(ascending=False).reset_index()
branches_rank[branches_rank['branch_id'].isin(
    set_of_unscaduled_branches)].head()

# %%

assert scheduled_df.shape[0] > 0
#%%
TEST_REPORT = []

# %%

schedued_pool_dates = master_df['p_date'].unique()
schedued_on_dates = scheduled_df['p_date'].unique()

schedued_pool_dates_count = master_df['p_date'].nunique()
schedued_on_dates_count = scheduled_df['p_date'].nunique()

TEST_REPORT.append(
    ('Descriptive', 'Date', 'schedued_pool_dates', schedued_pool_dates))
TEST_REPORT.append(
    ('Descriptive', 'Date', 'schedued_on_dates', schedued_on_dates))
TEST_REPORT.append(
    ('Descriptive', 'Date', 'schedued_pool_dates_count', schedued_pool_dates_count))
TEST_REPORT.append(
    ('Descriptive', 'Date', 'schedued_on_dates_count', schedued_on_dates_count))

# %%

number_of_branches = master_df['branch_id'].nunique()
number_of_restaurants = master_df['restaurant_id'].nunique()
number_of_areas = master_df['area_id'].nunique()
number_of_groups = master_df['group_id'].nunique()


TEST_REPORT.append(
    ('Descriptive', 'Stats', 'number_of_branches', number_of_branches))
TEST_REPORT.append(
    ('Descriptive', 'Stats', 'number_of_restaurants', number_of_restaurants))
TEST_REPORT.append(
    ('Descriptive', 'Stats', 'number_of_areas', number_of_areas))
TEST_REPORT.append(
    ('Descriptive', 'Stats', 'number_of_groups', number_of_groups))

# %%

number_of_essential_pairs = master_df[[
    'day', 'group_id', 'perday_resto']].drop_duplicates()['perday_resto'].sum()
schedued_branches_from_delivery_capacity = master_df[master_df['isScheduled']==1].shape[0]
schedued_branches_from_delivery_capacity_with_constraints = master_df[master_df['isScheduled']==1].shape[0]
schedued_branches_from_delivery_capacity_without_constraints = master_df.query('stat == "<<<SCHEDULED>>>"').shape[0]

valid_pairs_percentage = (schedued_branches_from_delivery_capacity / \
    number_of_essential_pairs * 100)

leftover_pairs_percentage = schedued_branches_from_delivery_capacity_without_constraints / \
    number_of_essential_pairs * 100

leftover_pairs_percentage = 100 - valid_pairs_percentage

after_leftover_pairs_percentage = schedued_branches_from_delivery_capacity / \
    number_of_essential_pairs * 100
missing_pairs_percentage = 100 - after_leftover_pairs_percentage

TEST_REPORT.append(('Comparative', 'Pairs', 'total_pairs_made',
                    schedued_branches_from_delivery_capacity))
TEST_REPORT.append(('Comparative', 'Pairs', 'valid_pairs_count',
                    schedued_branches_from_delivery_capacity_with_constraints))
TEST_REPORT.append(('Comparative', 'Pairs', 'leftover_pairs_count',
                    schedued_branches_from_delivery_capacity_without_constraints))
TEST_REPORT.append(('Comparative', 'Pairs', 'missing_pairs_count',
                    number_of_essential_pairs - schedued_branches_from_delivery_capacity))

TEST_REPORT.append(('Comparative', 'Pairs',
                    'supposedly_essential_pairs', number_of_essential_pairs))
TEST_REPORT.append(
    ('Comparative', 'Pairs', 'valid_pairs_percentage', valid_pairs_percentage))
TEST_REPORT.append(('Comparative', 'Pairs',
                    'leftover_pairs_percentage', leftover_pairs_percentage))
TEST_REPORT.append(('Comparative', 'Pairs',
                    'after_leftover_pairs_percentage', after_leftover_pairs_percentage))
TEST_REPORT.append(
    ('Comparative', 'Pairs', 'missing_pairs_percentage', missing_pairs_percentage))


#%%
# %%
def validate_perday_pairs(master_df):
    dropoff_perday_restorent = master_df.drop_duplicates(['p_date', 'group_id']).groupby(
        ['p_date', 'group_id'])['perday_resto'].sum().reset_index()
    scheduled_dropoff_perday_restorent = scheduled_df.query('schedule_status == 0').groupby(
        ['p_date', 'group_id'])['per_day_restaurants'].size().reset_index()
    scheduled_vs_poll_dropoff_perday_restorent = dropoff_perday_restorent.merge(
        scheduled_dropoff_perday_restorent, on=['p_date', 'group_id'], how='left')

    scheduled_vs_poll_dropoff_perday_restorent['per_day_restaurants'].fillna(
        0, inplace=True)
    scheduled_vs_poll_dropoff_perday_restorent['un_scheduled_count'] = scheduled_vs_poll_dropoff_perday_restorent.eval(
        'perday_resto - per_day_restaurants')

    # computing perday missing count in every catagory
    perday_vs_schedule_pairs = []
    for i in range(5, 2, -1):
        for j in range(i, -1, -1):
            t = scheduled_vs_poll_dropoff_perday_restorent.query(
                f'perday_resto == {i} and per_day_restaurants == {j}')
            t = t.groupby('p_date')['un_scheduled_count'].size(
            ).reset_index().assign(filter_on=f'{j}/{i}')
            perday_vs_schedule_pairs.append(t)

    perday_vs_schedule_pairs = pd.concat(perday_vs_schedule_pairs)
    perday_vs_schedule_pairs = perday_vs_schedule_pairs.pivot(
        index='p_date', columns='filter_on', values='un_scheduled_count')
    dd = perday_vs_schedule_pairs.sum(axis=0).to_frame().transpose()
    dd.index = ['All']
    perday_vs_schedule_pairs = perday_vs_schedule_pairs.append(dd)

    normalize_missing_counts = scheduled_vs_poll_dropoff_perday_restorent.groupby(
        ['p_date', 'un_scheduled_count'])['per_day_restaurants'].size().reset_index()

    normalize_missing_counts = normalize_missing_counts.pivot(
        index='p_date', columns='un_scheduled_count', values='per_day_restaurants')
    normalize_missing_counts = normalize_missing_counts.div(
        normalize_missing_counts.sum(axis=1), axis=0)
    new_columns_name = list(normalize_missing_counts.columns)
    new_columns_name = [f'#{int(i)}_pairs_missing%' for i in new_columns_name]
    normalize_missing_counts.columns = new_columns_name
    normalize_missing_counts = normalize_missing_counts.append(
        normalize_missing_counts.agg(['mean']).rename(index={'mean': 'All'}))

    perday_vs_schedule_pairs = pd.concat(
        [perday_vs_schedule_pairs, normalize_missing_counts], axis=1)

    return perday_vs_schedule_pairs


perday_vs_schedule_pairs = validate_perday_pairs(master_df)
perday_vs_schedule_pairs.to_csv(gapp(os.path.join(
    processed_base_data_path, 'output', 'validation', 'perday_vs_schedule_pairs.csv')))
TEST_REPORT.append(('Comparative', 'Pairs',
                    'perday_vs_schedule_pairs', '<<< perday_vs_schedule_pairs >>>'))
perday_vs_schedule_pairs
#%%

total_pairs_perday = master_df.query(
    'schedule_status==schedule_status').groupby('p_date')['orders'].size().tolist()
total_pairs_perday_without_voilation = master_df.query(
    'schedule_status == 0').groupby('p_date')['orders'].size().tolist()
sum_of_total_pairs = sum(total_pairs_perday)
sum_of_total_pairs_without_voilation = sum(
    total_pairs_perday_without_voilation)

TEST_REPORT.append(
    ('Comparative', 'Pairs', 'total_pairs_perday', total_pairs_perday))
TEST_REPORT.append(('Comparative', 'Pairs', 'total_pairs_perday_without_voilation',
                    total_pairs_perday_without_voilation))
TEST_REPORT.append(
    ('Comparative', 'Pairs', 'sum_of_total_pairs', sum_of_total_pairs))
TEST_REPORT.append(('Comparative', 'Pairs', 'sum_of_total_pairs_without_voilation',
                    sum_of_total_pairs_without_voilation))

scheduled_df.shape
#%%
# %%

total_orders_perday = master_df.query(
    'schedule_status==schedule_status').groupby(['p_date'])['orders'].sum().tolist()
total_orders_perday_without_voilation = master_df.query(
    'schedule_status == 0').groupby('p_date')['orders'].sum().tolist()
sum_of_total_orders = sum(total_orders_perday)
sum_of_total_orders_without_voilation = sum(
    total_orders_perday_without_voilation)

TEST_REPORT.append(
    ('Comparative', 'Orders', 'total_orders_perday', total_orders_perday))
TEST_REPORT.append(('Comparative', 'Orders', 'total_orders_perday_without_voilation',
                    total_orders_perday_without_voilation))
TEST_REPORT.append(
    ('Comparative', 'Orders', 'sum_of_total_orders', sum_of_total_orders))
TEST_REPORT.append(('Comparative', 'Orders', 'sum_of_total_orders_without_voilation',
                    sum_of_total_orders_without_voilation))
#%%
FLAG_BLACKOUT_TEST = True
if FLAG_BLACKOUT_TEST:
    blackout_constraint_voilate = scheduled_df.query(
        'is_not_blackout == 0 and schedule_status == 0')
    number_of_blackout_constraint_voilate = blackout_constraint_voilate.shape[0]

    TEST_REPORT.append(('Comparative', 'Blackout', 'is_any_blackout_constraint_voilate',
                        number_of_blackout_constraint_voilate != 0))
    TEST_REPORT.append(('Comparative', 'Blackout',
                        'number_of_blackout_constraint_voilate', number_of_blackout_constraint_voilate))
    blackout_constraint_voilate.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'blackout_rule_voilate_companies.csv')))
    TEST_REPORT.append(('Comparative', 'Blackout', 'blackout_rule_voilate_companies',
                        '<<< blackout_rule_voilate_companies >>>'))
#%%
scheduled_df.query('is_not_blackout == 0 and schedule_status == 0').shape

# %%

# scheduled_df.groupby(['p_date', 'branch_id'])['area_id'].apply(lambda x: x.unique()).reset_index()
multi_areas_branches = scheduled_df.groupby(['p_date', 'branch_id']).filter(lambda x: len(x['area_id'].unique()) > 1)
branches_breaching_cap = multi_areas_branches[multi_areas_branches['delivery_capacity'] == 1]
# is any branch_cap rule voilate
is_branch_cap_area_voilate = branches_breaching_cap[['p_date', 'branch_id', 'restaurant_id', 'area_id', 'delivery_capacity']].shape[0]
# total branches voilating cap rule
num_of_branches_cap_area_voilate = branches_breaching_cap[['p_date', 'branch_id', 'restaurant_id', 'area_id', 'delivery_capacity']].branch_id.nunique()
# to csv
branches_breaching_cap = branches_breaching_cap[['p_date', 'branch_id', 'restaurant_id', 'area_id', 'delivery_capacity']]

TEST_REPORT.append(('Comparative', 'Branch_Capacity_Rule', 'is_any_branch_cap_area_voilate',
                        is_branch_cap_area_voilate != 0))
TEST_REPORT.append(('Comparative', 'Branch_Capacity_Rule', 'number_of_branch_cap_area_voilate',
                        num_of_branches_cap_area_voilate))
TEST_REPORT.append(('Comparative', 'Branch_Capacity_Rule', 'number_of_branch_cap_area_voilate',
                        num_of_branches_cap_area_voilate))
branches_breaching_cap.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'branches_capacity_area_voilations.csv')), index=False)

TEST_REPORT.append(('Comparative', 'Branch_Capacity_Rule', 'branches_capacity_area_voilations',
                        '<<< branches_capacity_area_voilations >>>'))
#%%
# def validate_branchs_orders(master_df, TEST_REPORT):

#     # No more than 1 branch should be scheduled on a group in a single day
#     total_groups = scheduled_df.groupby(['p_date', 'branch_id'])['group_id'].count()
#     group_wise_df = total_groups.to_frame().reset_index()

#     # ATTENTION MANUAL RULE BREAK
#     ############################################################
#     # group_wise_df.loc[(group_wise_df.branch_id == 21),'group_id']=3
#     ############################################################


#     number_of_branch_capacity_rule_voilate = group_wise_df[group_wise_df['group_id'] > 1].shape[0]
#     temp_df = group_wise_df[group_wise_df['group_id'] > 1]



#     temp_df.to_csv(gapp(os.path.join(processed_base_data_path,
#                                   'output', 'validation', 'branch_capacity_rule_voilations.csv')), index=False)


#     TEST_REPORT.append(('Comparative', 'Branches',
#                         'is_any_branch_capacity_rule_voilate', number_of_branch_capacity_rule_voilate != 0))
#     TEST_REPORT.append(('Comparative', 'Branches',
#                         'number_of_branch_capacity_rule_voilate', number_of_branch_capacity_rule_voilate))
#     TEST_REPORT.append(('Comparative', 'Branches', 'branches_delivery_list_of_voilations_df',
#                         '<<< branches_delivery_list_of_voilations_df >>>'))


# validate_branchs_orders(master_df, TEST_REPORT)
#%%
def validate_cusine_rules(scheduled_df, cusine_table, TEST_REPORT):
    # check on resto types (6 atm)
    # check on cuisine types (16 atm)
    cusine_table = cusine_table.reset_index()
    cusine_table = pd.melt(cusine_table, id_vars='id', var_name="perday_resto",
                           value_name="max_scadule").rename(columns={'id': 'category'})
    cusine_table.fillna(100, inplace=True)
    cusine_table = cusine_table.astype(float)

    # filter out only weekdays schedules
    temp_scheduled_df = scheduled_df[scheduled_df['day'].isin([6, 0, 1, 2, 3])]

    cusin_rule_check = temp_scheduled_df.query('schedule_status == 0').groupby(['group_id', 'perday_resto', 'category'])[
        'orders'].size().reset_index().rename(columns={'orders': 'N_times_scheduled'})
    cusin_rule_check = cusin_rule_check.merge(
        cusine_table, on=['perday_resto', 'category'], how='left')

    # ATTENTION MANUAL RULE BREAK
    ############################################################
#     cusin_rule_check.loc[[2, 1220], 'N_times_scheduled'] = 10
    ############################################################

    cusin_rule_check['is_rule_break'] = cusin_rule_check.eval(
        'max_scadule < N_times_scheduled')

    temp = cusin_rule_check[cusin_rule_check['is_rule_break']]
    temp.to_csv(gapp(os.path.join(processed_base_data_path,
                                  'output', 'validation', 'cusin_level_rule_voilate.csv')))

    is_any_cusin_level_rule_voilate = temp.shape[0]
    # print(is_any_cusin_level_rule_voilate)
    # print(is_any_cusin_level_rule_voilate != 0)

    TEST_REPORT.append(('Comparative', 'Cusine', 'is_any_cusin_level_rule_voilate',
                        is_any_cusin_level_rule_voilate != 0))
    TEST_REPORT.append(('Comparative', 'Cusine',
                        '#of_cusin_level_rule_voilate', is_any_cusin_level_rule_voilate))
    TEST_REPORT.append(('Comparative', 'Cusine',
                        'cusin_level_rule_voilate', '<<< cusin_level_rule_voilate >>>'))
#%%
validate_cusine_rules(scheduled_df, cusine_table, TEST_REPORT)
#%%
def validate_resto_type(scheduled_df, TEST_REPORT):
    # find any duplicate types in the DF for a given day and dropoff
    # TODO add schedule_status == 0 here
    # Use keep =false to display all the dupes
    t_scheduled_df = scheduled_df.loc[scheduled_df['schedule_status'] == 0.0]
    t_scheduled_df['duplicate_type'] = t_scheduled_df.duplicated(
        subset=['p_date', 'group_id', 'type'], keep="first")
    # t_scheduled_df['duplicate_type'] = t_scheduled_df.duplicated(subset=['p_date','client_id', 'restaurant_type'], keep=False)
    temp_scheduled_df = t_scheduled_df.loc[t_scheduled_df['duplicate_type'] == True]
    temp_scheduled_df = temp_scheduled_df[['p_date', 'group_id', 'duplicate_type',
                                           'branch_id', 'restaurant_id', 'type', 'restaurant_type_name']]

    temp_scheduled_df.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'resto_type_duplicate.csv')))
    is_any_resto_type_duplicate = temp_scheduled_df.empty
    no_of_type_dupe = temp_scheduled_df.shape[0]
    TEST_REPORT.append(('Comparative', 'Restaurant Type Duplicate',
                        'is_any_resto_type_duplicate', not is_any_resto_type_duplicate))
    TEST_REPORT.append(('Comparative', 'Restaurant Type Duplicate',
                        '#of_resto_type_duplicate', no_of_type_dupe))
    TEST_REPORT.append(('Comparative', 'Restaurant Type Duplicate',
                        'resto_type_duplicate', '<<< resto_type_duplicate >>>'))
#%%
# validate per day resto type dupe
validate_resto_type(scheduled_df, TEST_REPORT)
# %%
import numpy as np
scheduled_df['p_date'] = pd.to_datetime(scheduled_df['p_date'])
pdr_type_conditions = [
    scheduled_df['p_date'].dt.day_name().isin(
        ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])
]

choices = [0]
scheduled_df['pdr_type_key'] = np.select(
    pdr_type_conditions, choices, default=None)
#%%
pdr_type_dct = {
    0: {
        3: {
            (1,): 1,
            (9,): 1,
            (5, 6, 7): 1
        },
        4: {
            (1,): 1,
            (9,): 1,
            (5, 6, 7): 2
        },
        5: {
            (1,): 1,
            (9,): 1,
            (5,): 1,
            (6,): 1,
            (7,): 1       
        }
    }
}

# %%
def validate_type_voilate(scheduled_df, pdr_type_dct, TEST_REPORT):
    data = []
    type_voilate = []
    temp_scheduled_df = scheduled_df.groupby(['p_date',  'per_day_restaurants', 'group_id', 'pdr_type_key'])[
        'type'].apply(list).reset_index()
    for index, row in temp_scheduled_df.iterrows():
        if row["per_day_restaurants"] != len(row["type"]):
            data.append([row["p_date"], row["group_id"],
                         row["per_day_restaurants"], row["type"]])
        curr_dct = pdr_type_dct[row["pdr_type_key"]].get(
            row["per_day_restaurants"], None)
        if curr_dct:
            for k, v in curr_dct.items():
                if len(set(row['type']) & set(k)) > v:
                    type_voilate.append([row["p_date"], row["group_id"],
                                         row["per_day_restaurants"], row["type"], k, v])

    # Create the pandas DataFrame
    type_mismatch_df = pd.DataFrame(data, columns=[
                                    'date', 'group_id', 'per_day_restaurants', 'scheduled_restaurant_types'])

    # Create the pandas DataFrame
    type_voilate_df = pd.DataFrame(type_voilate, columns=[
                                   'date', 'group_id', 'per_day_restaurants', 'scheduled_restaurant_types', 'rule_voilate_types', 'max_to_be_scheduled'])

    # print(type_mismatch_df)
    type_mismatch_df.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'resto_type_mismatch.csv')))
    type_voilate_df.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'resto_type_voilate.csv')))

    is_any_resto_type_mismatch = type_mismatch_df.empty
    no_of_type_mismatch = type_mismatch_df.shape[0]
    # TEST_REPORT.append(('Comparative', 'Restaurant Type Mismatch', 'is_any_resto_type_mismatch', not is_any_resto_type_mismatch))
    # TEST_REPORT.append(('Comparative', 'Restaurant Type Mismatch', '#of_resto_type_mismatch', no_of_type_mismatch))
    # TEST_REPORT.append(('Comparative', 'Restaurant Type Mismatch', 'resto_type_mismatch', '<<< resto_type_mismatch >>>'))

    is_any_resto_type_voilate = type_voilate_df.empty
    no_of_type_voilate = type_voilate_df.shape[0]
    TEST_REPORT.append(('Comparative', 'Restaurant Type Voilate',
                        'is_any_resto_type_voilate', not is_any_resto_type_voilate))
    TEST_REPORT.append(('Comparative', 'Restaurant Type Voilate',
                        '#of_resto_type_voilate', no_of_type_voilate))
    TEST_REPORT.append(('Comparative', 'Restaurant Type Voilate',
                        'resto_type_voilate', '<<< resto_type_voilate >>>'))


# %%
validate_type_voilate(scheduled_df, pdr_type_dct, TEST_REPORT)
#%%
# validate restos with delivery cap = 2
#%%
# %%
def exp_top_resto():
    orders_df = pd.read_csv(gap('orders.csv'),
                            usecols=['schedule_on', 'company_id', 'quantity', 'category_id', 'branch_id', 'deleted_at'],
                            parse_dates=['schedule_on'])
    orders_df = orders_df[orders_df['deleted_at'].isna()]
    orders_df = orders_df[orders_df['category_id'] == 2]
    # START_REFFERENCE_WINDOW_DATE = END_REFFERENCE_WINDOW_DATE
    orders_df = orders_df[orders_df['schedule_on'].between(START_REFFERENCE_WINDOW_DATE, END_REFFERENCE_WINDOW_DATE)]

    brans_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id']).rename(columns={'id': 'branch_id'})
    orders_df = orders_df.merge(brans_df, on='branch_id', how='left')

    comp_df = pd.read_csv(gap('companies.csv'), usecols=['id', 'group_id']).rename(columns={'id': 'company_id'})
    orders_df = orders_df.merge(comp_df, on='company_id', how='inner')

    orders_df = orders_df.groupby(['schedule_on', 'group_id', 'restaurant_id'])['quantity'].sum().reset_index()

    menu_schedules_df = pd.read_csv(gap('menu_schedules.csv'),
                                    usecols=['company_id', 'schedule_on', 'branch_id', 'deleted_at'],
                                    parse_dates=['schedule_on'])
    menu_schedules_df = menu_schedules_df[
        menu_schedules_df['schedule_on'].between(START_REFFERENCE_WINDOW_DATE, END_REFFERENCE_WINDOW_DATE)]
    menu_schedules_df = menu_schedules_df[menu_schedules_df['deleted_at'].isna()]
    _ = menu_schedules_df.pop('deleted_at')
    menu_schedules_df = menu_schedules_df[['company_id', 'schedule_on', 'branch_id']].drop_duplicates(
        subset=['company_id', 'branch_id'], keep='last')
    print('Checking menu_schedules table size :: ', menu_schedules_df.shape)
    menu_schedules_df.reset_index(drop=True, inplace=True)
    menu_schedules_df.rename(columns={'schedule_on': 'last_schedule_on'}, inplace=True)

    menu_schedules_df = menu_schedules_df.merge(comp_df, on='company_id', how='inner')
    menu_schedules_df = menu_schedules_df.merge(brans_df, on='branch_id', how='left')

    menu_schedules_df.rename(columns={'last_schedule_on': 'schedule_on'}, inplace=True)
    menu_schedules_df = menu_schedules_df[['schedule_on', 'group_id', 'restaurant_id']].drop_duplicates()

    orders_df = orders_df.merge(menu_schedules_df, on=['schedule_on', 'group_id', 'restaurant_id'], how='right')
    orders_df['quantity'] = orders_df['quantity'].fillna(0)

    orders_df.sort_values('schedule_on', ascending=True, inplace=True)
    orders_df['sch_position'] = orders_df.groupby(['group_id', 'restaurant_id'])['schedule_on'].rank(ascending=True)

    orders_df['weighted_orders'] = orders_df['quantity'] * orders_df['sch_position']
    orders_df = orders_df.groupby(['group_id', 'restaurant_id'])['weighted_orders'].mean().reset_index()
    orders_df.rename(columns={'weighted_orders': 'quantity'}, inplace=True)
    orders_df = orders_df.sort_values('quantity', ascending=False)
    orders_df = orders_df.groupby(['group_id']).head(5)

    tc_list_df = orders_df.groupby('group_id')['restaurant_id'].apply(list).reset_index()
    tc_list_df = tc_list_df.rename(columns={'restaurant_id': 'top_restos'})

    return tc_list_df


def top_restaurant_voilations(top_resto_list_df, TEST_REPORT):
    active_df = scheduled_df.merge(top_resto_list_df, on='group_id', how='left')
    active_df.to_csv('complete.csv', index=False)
    top_resto_df = active_df.groupby(['p_date', 'group_id'],as_index=False)['restaurant_id'].agg({'restaurant_id': lambda x: list(x)})
    temp_df = active_df[['p_date', 'group_id', 'top_restos']]
    top_resto_df = top_resto_df.merge(temp_df, on=['p_date', 'group_id'], how='left')

    top_resto_df = top_resto_df[~top_resto_df.astype(str).duplicated()]

    # drop rows without top_restos
    top_resto_df = top_resto_df[~top_resto_df['top_restos'].isnull()]

    # get intersection
    top_resto_df['scheduled_top_restos'] = [list(set(a).intersection(b)) for a, b in zip(top_resto_df.restaurant_id, top_resto_df.top_restos)]

    top_resto_df['scheduled_top_resto_count'] = [len(set(a).intersection(b)) for a, b in zip(top_resto_df.restaurant_id, top_resto_df.top_restos)]

    top_resto_df = top_resto_df[top_resto_df['scheduled_top_resto_count'] > 2]

    top_resto_df.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'top_resto_voilations.csv')), index=False)

    is_any_top_resto_voilate = top_resto_df.empty
    no_of_top_resto_voilate = top_resto_df.shape[0]
    TEST_REPORT.append(('Comparative', 'Top Restaurant Voilate',
                        'is_any_top_resto_voilate', not is_any_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'Top Restaurant Voilate',
                        '#no_of_top_resto_voilate', no_of_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'Top Restaurant Voilate',
                        'resto_type_voilate', '<<< top_resto_voilations >>>'))


#%%
top_resto_list_df = exp_top_resto()
top_restaurant_voilations(top_resto_list_df, TEST_REPORT)
#%%
result = pd.DataFrame(TEST_REPORT, columns=[
                      'Scope', 'Domain', 'Test Name', 'Value']).set_index(['Scope', 'Domain'])
result.to_csv(gapp(os.path.join(processed_base_data_path,
                                'output', 'validation', 'validation_result.csv')))
# print(result)
# exit()
# #%%

# def filter_top_n_restos():
#     # Top N restaurants
#     # generic method:

#     # take 3 month orders table data
#     # group_id > resto with max orders (50 or more less)
#     # take top 5

#     # check no more than 2 top should be present in one-day master_df for one group

#     orders_df = pd.read_csv(gap('orders.csv'), parse_dates=['schedule_on'])
#     # trim orders_df data x3 months backward from given input date
#     # start_date = '2021-04-04'
#     start_date = args.prediction_start_date
#     start_date = datetime.fromisoformat(start_date)
#     end_date = start_date + relativedelta(months=-3)

#     orders_df = orders_df[(orders_df['schedule_on'] <= start_date) & (orders_df['schedule_on'] >= end_date) & (orders_df['status']==5)]
#     #%%
#     # get group_id
#     orders_df = orders_df[orders_df['status']==5]
#     orders_df = orders_df.merge(companies_df, on='company_id', how='left')

#     # merge orders data with branches
#     orders_df = orders_df.merge(branchs_df, on='branch_id', how='left')
#     #%%
#     orders_df = orders_df.groupby(['branch_id', 'group_id', 'restaurant_id'])['quantity'].sum().reset_index()

#     #%%
#     # get top 5 restos by orders in a group

#     orders_df = orders_df.sort_values('quantity',ascending = False).groupby('group_id').head(5)


#     merge schedule_df with latest orders_df (left join on group_id)
#     groupby_ on latest merged df and create another column based on common restos (True/False)



#%%