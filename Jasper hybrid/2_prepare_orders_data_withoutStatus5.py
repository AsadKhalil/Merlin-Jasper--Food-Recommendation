# Running command:
# python3 1_prepare_orders_data_withoutStatus5.py --prediction_start_date "2021-02-14" 
# --prediction_end_date "2021-02-20" --data_dir "/home/munchon/Documents/Munir/Documents/merlin_py/data" --run_id 'local'
#%%
import argparse
import datetime
import logging
import sys
import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--all_prepare_start_date", help="Actual start date from which order prepare script start. This "
                                                     "will override the --training_start_date parameter.")
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.", required=True)

args = parser.parse_args()
if args.all_prepare_start_date is not None:
    all_prepare_start_date = args.all_prepare_start_date
    START_DATE = datetime.strptime(all_prepare_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')

else:
    prediction_start_date = args.prediction_start_date
    PREDICTION_START_DATE = prediction_start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
    prediction_start_date = prediction_start_date + relativedelta(months=-6)
    weekday = prediction_start_date.weekday() + 1
    week_day_diff = weekday % 7
    prediction_start_date = prediction_start_date + relativedelta(days= (week_day_diff * -1))
    START_DATE = prediction_start_date.strftime('%Y-%m-%d')

#%%
data_base_path = args.data_dir
run_id = args.run_id
log_dir_path = os.path.join(data_base_path, '..', 'logs')
Path(log_dir_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig( level=logging.DEBUG,
    format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt = '%Y-%m-%d:%H:%M:%S',
    handlers = [ logging.FileHandler(Path(log_dir_path).joinpath(run_id+'.log')), logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger(__name__)
logger.debug(START_DATE)
logger.debug(args.data_dir)
logger.debug(args.run_id)

#%%
# data_base_path = 'C:\\work\\marlin_scheduling\\data'
raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
#%%

#%%
# START_DATE = '2020-06-01'
# building_schedule_df = pd.read_csv(gap('menu_schedules_building.csv'), usecols=['menu_id', 'building_id', 'schedule_on', 'branch_id', 'deleted_at'], parse_dates=['schedule_on'])
# building_schedule_df = building_schedule_df.loc[building_schedule_df['schedule_on'] >= START_DATE]
companies_schedule_df = pd.read_csv(gap('menu_schedules.csv'), usecols=['menu_id', 'company_id', 'schedule_on', 'branch_id', 'deleted_at'], parse_dates=['schedule_on'])
companies_schedule_df = companies_schedule_df.loc[companies_schedule_df['schedule_on'] >= START_DATE]

# logger.info('building schedule size: %s', building_schedule_df.shape)
# building_schedule_df = building_schedule_df[building_schedule_df['deleted_at'].isna()]
# logger.info('After droping deleted records building schedule size: %s', building_schedule_df.shape)

logger.info('companies schedule size: %s', companies_schedule_df.shape)
companies_schedule_df = companies_schedule_df[companies_schedule_df['deleted_at'].isna()]
logger.info('After droping deleted records companies schedule size: %s', companies_schedule_df.shape)
# companies_schedule_df = companies_schedule_df.drop_duplicates(['company_id', 'schedule_on', 'branch_id'], keep='last')
# print('After droping duplicate records companies schedule size: ', companies_schedule_df.shape)
#%%

menus_df = pd.read_csv(gap('menus.csv'), usecols=['id', 'category_id'])
menus_df.rename(columns={'id': 'menu_id'}, inplace=True)
menus_df = menus_df.loc[menus_df['category_id'] == 2]

# building_schedule_df = building_schedule_df[building_schedule_df['menu_id'].isin(menus_df['menu_id'])]
# logger.info('After dropping non-lunch records buildings schedule size: %s', building_schedule_df.shape)
companies_schedule_df = companies_schedule_df[companies_schedule_df['menu_id'].isin(menus_df['menu_id'])]
logger.info('After dropping non-lunch records companies schedule size: %s', companies_schedule_df.shape)
# del(menus_df)

#%%
# building_schedule_df = building_schedule_df.drop_duplicates(['building_id', 'schedule_on', 'branch_id'], keep='last')
# logger.info('After droping duplicate records building schedule size: %s', building_schedule_df.shape)

companies_schedule_df = companies_schedule_df.drop_duplicates(['company_id', 'schedule_on', 'branch_id'], keep='last')
logger.info('After droping duplicate records companies schedule size: %s', companies_schedule_df.shape)

#%%
branch_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id', 'name','status']).rename(columns={'status':'branch_status'})
branch_df = branch_df[(branch_df['branch_status'] == 1) | (branch_df['branch_status'] == 2)]
branch_df = branch_df[~branch_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)].drop('name', axis=1)
branch_df.rename(columns={'id':'branch_id'}, inplace=True)
branch_df.drop(['branch_status'], axis=1, inplace=True)

companies_schedule_df = companies_schedule_df.merge(branch_df, on='branch_id', how='inner')
companies_schedule_df.drop(['branch_id', 'menu_id', 'deleted_at'], axis=1, inplace=True)

# building_schedule_df = building_schedule_df.merge(branch_df, on='branch_id', how='inner')
# building_schedule_df.drop(['branch_id', 'menu_id', 'deleted_at'], axis=1, inplace=True)

#%%
# building_schedule_df = building_schedule_df.drop_duplicates(['building_id', 'schedule_on', 'restaurant_id'], keep='last')
# logger.info('After droping duplicate records building schedule size: %s', building_schedule_df.shape)

companies_schedule_df = companies_schedule_df.drop_duplicates(['company_id', 'schedule_on', 'restaurant_id'], keep='last')
logger.info('After droping duplicate records companies schedule size: %s', companies_schedule_df.shape)

#%%
usecols=['id', 'quantity', 'category_id', 'branch_id', 'status', 'schedule_on', 'company_id'] # removing building_id
orders_df = pd.read_csv(gap('orders.csv'), usecols=usecols, parse_dates=['schedule_on'])
orders_df.rename(columns={'id': 'order_id'}, inplace=True)
orders_df = orders_df.loc[orders_df['schedule_on'] >= START_DATE]
orders_df = orders_df.loc[orders_df['category_id'] == 2]

#%%
test_companies_df = pd.read_csv(gap('companies.csv'), usecols=['id', 'name'])
test_companies_df = test_companies_df[test_companies_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)]['id']
# test_buildings_df = pd.read_csv(gap('buildings.csv'), usecols=['id', 'name'])
# test_buildings_df = test_buildings_df[test_buildings_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)]['id']

orders_df = orders_df[(~orders_df['company_id'].isin(test_companies_df))]
# orders_df = orders_df[(~orders_df['building_id'].isin(test_buildings_df))].copy()

#%%
ratings_df = pd.read_csv(gap('meal_ratings.csv'), usecols=['order_id', 'rating'])
orders_df = orders_df.merge(ratings_df, how='left', on='order_id')
orders_df = orders_df.merge(branch_df, on='branch_id', how='left')
# orders_df = orders_df.query(' status == 5')

generate_client_ids = lambda x: str(int(x['company_id'])) + '_C' # "removing extra func arguments:" if pd.isna(x['building_id']) else str(int(x['building_id'])) + '_B'
orders_df['client_id'] = orders_df.apply(generate_client_ids, axis=1)
# orders_df.drop(['branch_id', 'order_id', 'category_id'], axis=1, inplace=True)

#%%
orders_df = orders_df.groupby(['schedule_on', 'client_id', 'restaurant_id']).agg({'rating':'mean', 'quantity': 'sum'}).reset_index()

#%%
# building_schedule_df['client_id'] = building_schedule_df['building_id'].astype(int).astype(str)+'_B'
companies_schedule_df['client_id'] = companies_schedule_df['company_id'].astype(int).astype(str)+'_C'

temp_df = companies_schedule_df.copy()[['schedule_on', 'restaurant_id', 'client_id']]

#%%
temp_df = temp_df.merge(orders_df, on=['schedule_on', 'restaurant_id', 'client_id'], how='left')
temp_df['quantity'] = temp_df['quantity'].fillna(0)


#%%
# Drop clients who didn't place any order from last N months
def get_resent_active_clients(df, months=-3, vital_client_window_end_date=PREDICTION_START_DATE):
    vital_client_window_start_date = vital_client_window_end_date + relativedelta(months=months)
    vital_clients = temp_df[temp_df['schedule_on'].between(vital_client_window_start_date, vital_client_window_end_date)]
    vital_clients = vital_clients[['client_id', 'quantity']].groupby('client_id')['quantity'].sum()
    vital_clients_list = vital_clients[vital_clients > 0].index

    logger.info('Select window for vital clients from %s -> %s', vital_client_window_start_date , vital_client_window_end_date)

    logger.info('Total clients: %s', df['client_id'].nunique())
    logger.info('Total clients in vital window: %s', vital_clients.shape[0])
    logger.info('Total vital clients : %d', len(vital_clients_list))

    return vital_clients_list

vital_clients_list = get_resent_active_clients(temp_df, months=-2, vital_client_window_end_date=PREDICTION_START_DATE)

temp_df = temp_df[temp_df['client_id'].isin(vital_clients_list)]

#%%
temp_df.to_csv(gapp('all_prepared_scedules_vs_orders_with_rating_Qcount.csv'), index=False)

#%%
logger.info('Final df size: %s', temp_df.shape)
logger.info('Final df size after drop duplicates: %s', temp_df.drop_duplicates(['client_id', 'restaurant_id', 'schedule_on']).shape)