
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import kurtosis, skew
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--prediction_end_date", help="End date of orders prediction from.", required=True)
parser.add_argument("--force_start_date", help="Actual start date from which order prepare script start. This "
                                                     "will override the --training_start_date parameter.")
parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.", required=True)
parser.add_argument("--training_max_date", help="Maximum date include in training data.", required=True)

args = parser.parse_args()

training_max_date = args.training_max_date
prediction_start_date = None
start_date = None
end_date = None
prediction_end_date = args.prediction_end_date
if args.force_start_date is not None:
    force_start_date = args.force_start_date
    force_start_date = datetime.strptime(force_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    print(force_start_date)
    start_date = force_start_date
else:
    prediction_start_date = args.prediction_start_date
    prediction_start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
    start_date = prediction_start_date + relativedelta(months=-5)
    weekday = start_date.weekday() + 1
    week_day_diff = weekday % 7
    start_date = start_date + relativedelta(days = (week_day_diff * -1))
    start_date = start_date.strftime('%Y-%m-%d')
    print(start_date)

# end_date = datetime.strptime(args.prediction_end_date, '%Y-%m-%d')
# end_date = (prediction_start_date + relativedelta(days=-1)).strftime('%Y-%m-%d')
training_max_date = datetime.strptime(training_max_date, '%Y-%m-%d')
end_date = (training_max_date + relativedelta(days=0)).strftime('%Y-%m-%d')
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

logger.debug('--prediction_start_date = %s', prediction_start_date)
logger.debug('--prediction_end_date = %s', prediction_end_date)
logger.debug('--data_dir = %s', data_base_path)
logger.debug('--start_date = %s', start_date)
logger.debug('--end_date = %s', end_date)
logger.debug('--run_id = %s', run_id)
#%%
# assert False
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

start_time = time.time()
# %%
# start_date = '2020-07-05'
# end_date = '2020-10-17'

# data_base_path = 'C:\\work\\marlin_scheduling\\data'
raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
temporary_object_store_path = os.path.join(processed_base_data_path, 'pickles')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)
Path(temporary_object_store_path).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
gapt = lambda x: os.path.join(temporary_object_store_path, x)
# %%
c2r_sched_vs_orders_df = pd.read_csv(gapp('all_prepared_scedules_vs_orders_with_rating_Qcount.csv'),
                                     parse_dates=['schedule_on'])
c2r_sched_vs_orders_df.rename(columns={'quantity': 'orders'}, inplace=True)

def remove_scheduleon_holidays(df):

    df = df[df['schedule_on'] != '2020-07-30']
    df = df[df['schedule_on'] != '2020-08-02']
    df = df[df['schedule_on'] != '2020-08-23']
    df = df[df['schedule_on'].dt.month != 12]
    df = df[df['schedule_on'] != '2021-01-01']
    df = df[df['schedule_on'] != '2020-01-02']
    return df

c2r_sched_vs_orders_df = remove_scheduleon_holidays(c2r_sched_vs_orders_df)

#%%
c2r_sched_vs_orders_df = c2r_sched_vs_orders_df.loc[c2r_sched_vs_orders_df['schedule_on'].between(start_date, end_date)]
c2r_sched_vs_orders_df['schedule_on'] = c2r_sched_vs_orders_df['schedule_on'] + pd.Timedelta(days=1)

#c2r_sched_vs_orders_df['week_no'] = c2r_sched_vs_orders_df['schedule_on'].dt.weekofyear
c2r_sched_vs_orders_df['iso_calender'] = c2r_sched_vs_orders_df['schedule_on'].dt.strftime('%G%V')
c2r_sched_vs_orders_df["iso_calender"] = c2r_sched_vs_orders_df["iso_calender"].astype(str).astype(int)
weeks = c2r_sched_vs_orders_df.groupby(['iso_calender']).size().reset_index(name='counts')
weeks.drop('counts', axis=1, inplace=True)
weeks['iso_calender'] = weeks['iso_calender'].sort_values(ascending=True)
weeks['week_no'] = weeks.index + 1
c2r_sched_vs_orders_df = c2r_sched_vs_orders_df.merge(weeks)
c2r_sched_vs_orders_df.drop('iso_calender', axis=1, inplace=True)

logger.info('At stage # %d', 1)

#%%


from multiprocessing import Pool, cpu_count

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def platform_specific_parallelism(dfGroup, func):
    print(sys.platform)
    print(cpu_count())
    if sys.platform == 'win32':
        return dfGroup.apply(func)
    else:
        return applyParallel(dfGroup, func)


# %%
c_stats_df = c2r_sched_vs_orders_df.groupby(['client_id', 'week_no']).agg(
    {'rating': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew]})
c_stats_df.columns = ['_'.join(i) for i in c_stats_df.columns]
c_stats_df = c_stats_df.reset_index()

c_stats_expanding_df = c2r_sched_vs_orders_df.groupby(['client_id', 'week_no'])[
    ['rating', 'orders']].mean().reset_index().sort_values('week_no')

c_stats_expanding_df = c_stats_expanding_df.groupby(['client_id']).expanding().agg(
    {'rating': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'week_no': 'max'})
c_stats_expanding_df.columns = ['exp_' + '_'.join(i) for i in c_stats_expanding_df.columns]
c_stats_expanding_df.reset_index(inplace=True)
c_stats_expanding_df.rename(columns={'exp_week_no_max': 'week_no'}, inplace=True)
c_stats_expanding_df.drop('level_1', axis=1, inplace=True)

c_all_stats_df = c_stats_df.merge(c_stats_expanding_df, how='left', on=['client_id', 'week_no'])

c_all_stats_df['is_first_time_company_schedule'] = c_all_stats_df.groupby('client_id')['week_no'].transform(min)
c_all_stats_df['is_first_time_company_schedule'] = c_all_stats_df['is_first_time_company_schedule'] == c_all_stats_df[
    'week_no']

MAX_WEEK_NO = c_all_stats_df['week_no'].max()


def complete_list_seq(x):
    min_val = min(x)
    max_val = MAX_WEEK_NO
    return list(range(min_val, max_val + 1))


c_all_stats_hooks_df = c_all_stats_df.groupby(['client_id'])['week_no'].aggregate(
    lambda x: complete_list_seq(x)).reset_index()
c_all_stats_hooks_df = c_all_stats_hooks_df.explode('week_no')

c_all_stats_df = c_all_stats_hooks_df.merge(c_all_stats_df, on=['client_id', 'week_no'], how='outer').sort_values(
    'week_no')

def c_ffill_bfill(x):
    bfill_columns = ['is_first_time_company_schedule']
    afill_columns = x.columns.difference(bfill_columns)

    x[bfill_columns] = x[bfill_columns].shift(-1)

    x.loc[:, bfill_columns] = x.loc[:, bfill_columns].bfill()
    x.loc[:, afill_columns] = x.loc[:, afill_columns].ffill()

    x['is_first_time_company_schedule'] = x['is_first_time_company_schedule'].fillna(False)

    return x

# c_all_stats_df = c_all_stats_df.groupby(['client_id'], as_index=False).apply(c_ffill_bfill)

c_all_stats_df = platform_specific_parallelism(c_all_stats_df.groupby(['client_id'], as_index=False), c_ffill_bfill)

logger.info('At stage # %d', 2)
# %%
r_stats_df = c2r_sched_vs_orders_df.groupby(['restaurant_id', 'week_no']).agg(
    {'rating': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew]})
r_stats_df.columns = ['_'.join(i) for i in r_stats_df.columns]
r_stats_df = r_stats_df.reset_index()

r_stats_expanding_df = c2r_sched_vs_orders_df.groupby(['restaurant_id', 'week_no'])[
    ['rating', 'orders']].mean().reset_index().sort_values('week_no')

r_stats_expanding_df = r_stats_expanding_df.groupby(['restaurant_id']).expanding().agg(
    {'rating': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'week_no': 'max'})
r_stats_expanding_df.columns = ['exp_' + '_'.join(i) for i in r_stats_expanding_df.columns]
r_stats_expanding_df.reset_index(inplace=True)
r_stats_expanding_df.rename(columns={'exp_week_no_max': 'week_no'}, inplace=True)
r_stats_expanding_df.drop('level_1', axis=1, inplace=True)

r_all_stats_df = r_stats_df.merge(r_stats_expanding_df, how='left', on=['restaurant_id', 'week_no'])

r_all_stats_df['is_first_time_restaurant_schedule'] = r_all_stats_df.groupby('restaurant_id')['week_no'].transform(min)
r_all_stats_df['is_first_time_restaurant_schedule'] = r_all_stats_df['is_first_time_restaurant_schedule'] == \
                                                      r_all_stats_df['week_no']

MAX_WEEK_NO = r_all_stats_df['week_no'].max()


def complete_list_seq(x):
    min_val = min(x)
    max_val = MAX_WEEK_NO
    return list(range(min_val, max_val + 1))


r_all_stats_hooks_df = r_all_stats_df.groupby(['restaurant_id'])['week_no'].aggregate(
    lambda x: complete_list_seq(x)).reset_index()
r_all_stats_hooks_df = r_all_stats_hooks_df.explode('week_no')

r_all_stats_df = r_all_stats_hooks_df.merge(r_all_stats_df, on=['restaurant_id', 'week_no'], how='outer').sort_values(
    'week_no')

def r_ffill_bfill(x):
    bfill_columns = ['is_first_time_restaurant_schedule']
    afill_columns = x.columns.difference(bfill_columns)

    x[bfill_columns] = x[bfill_columns].shift(-1)

    x.loc[:, bfill_columns] = x.loc[:, bfill_columns].bfill()
    x.loc[:, afill_columns] = x.loc[:, afill_columns].ffill()

    x['is_first_time_restaurant_schedule'] = x['is_first_time_restaurant_schedule'].fillna(False)

    return x


r_all_stats_df = platform_specific_parallelism(r_all_stats_df.groupby(['restaurant_id'], as_index=False), r_ffill_bfill)
logger.info('At stage # %d', 3)
# %%

c2r_stats_df = c2r_sched_vs_orders_df.groupby(['restaurant_id', 'client_id', 'week_no']).agg(
    {'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew]})

c2r_stats_df.columns = ['_'.join(i) for i in c2r_stats_df.columns]
c2r_stats_df = c2r_stats_df.reset_index()

c2r_stats_expanding_df = c2r_sched_vs_orders_df.groupby(['restaurant_id', 'client_id', 'week_no'])[
    ['orders']].mean().reset_index().sort_values('week_no')

c2r_stats_expanding_df = c2r_stats_expanding_df.groupby(['restaurant_id', 'client_id']).expanding().agg(
    {'orders': [np.min, np.max, np.mean, np.median, np.std, kurtosis, skew],
     'week_no': 'max'})
c2r_stats_expanding_df.columns = ['exp_' + '_'.join(i) for i in c2r_stats_expanding_df.columns]
c2r_stats_expanding_df.reset_index(inplace=True)
c2r_stats_expanding_df.rename(columns={'exp_week_no_max': 'week_no'}, inplace=True)

c2r_all_stats_df = c2r_stats_df.merge(c2r_stats_expanding_df, how='left', on=['restaurant_id', 'client_id', 'week_no'])

c2r_all_stats_df['relative_prograssinve_schedule_rank'] = c2r_all_stats_df.groupby(['restaurant_id', 'client_id'])[
    'week_no'].rank("dense", ascending=True)
c2r_all_stats_df['is_first_time_c2r'] = c2r_all_stats_df['relative_prograssinve_schedule_rank'] == 1

MAX_WEEK_NO = c2r_all_stats_df['week_no'].max()


def complete_list_seq(x):
    min_val = min(x)
    max_val = MAX_WEEK_NO
    return list(range(min_val, max_val + 1))

c2r_hooks_df = c2r_all_stats_df.groupby(['restaurant_id', 'client_id'])['week_no'].aggregate(
    lambda x: complete_list_seq(x)).reset_index()
c2r_hooks_df = c2r_hooks_df.explode('week_no')

c2r_all_stats_df = c2r_all_stats_df.merge(c2r_hooks_df, on=['restaurant_id', 'client_id', 'week_no'],
                                          how='outer').sort_values('week_no')

def ffill_bfill(x):
    bfill_columns = ['is_first_time_c2r', 'relative_prograssinve_schedule_rank']
    afill_columns = x.columns.difference(bfill_columns)

    x[bfill_columns] = x[bfill_columns].shift(-1)

    x.loc[:, bfill_columns] = x.loc[:, bfill_columns].bfill()
    x.loc[:, afill_columns] = x.loc[:, afill_columns].ffill()
    max_relative_progress = x['relative_prograssinve_schedule_rank'].max()
    x['relative_prograssinve_schedule_rank'] = x['relative_prograssinve_schedule_rank'].fillna(
        max_relative_progress + 1)
    x['is_first_time_c2r'] = x['is_first_time_c2r'].fillna(False)

    return x


# c2r_all_stats_df = c2r_all_stats_df.groupby(['restaurant_id', 'client_id'], as_index=False).apply(ffill_bfill)
c2r_all_stats_df = platform_specific_parallelism(c2r_all_stats_df.groupby(['restaurant_id', 'client_id'], as_index=False), ffill_bfill)

del (c2r_stats_df, c2r_stats_expanding_df, c2r_hooks_df)
logger.info('At stage # %d', 4)
# %%

c_users_df = pd.read_csv(gap('orders.csv'),
                         usecols=['user_id', 'company_id', 'schedule_on', 'user_subscription_id', 'category_id',
                                  'building_id', 'branch_id', 'quantity', 'status'])
c_users_df['schedule_on'] = pd.to_datetime(c_users_df['schedule_on'])
c_users_df = remove_scheduleon_holidays(c_users_df)
c_users_df = c_users_df.loc[
    (c_users_df['schedule_on'].between(start_date, end_date)) & (c_users_df['category_id'] == 2) & (
            c_users_df['status'] == 5)]
c_users_df['user_subscription_id'] = ~c_users_df['user_subscription_id'].isnull()

branch_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id'])
branch_df.rename(columns={'id': 'branch_id'}, inplace=True)

c_users_df = c_users_df.merge(branch_df, on='branch_id', how='left')
c_users_df.drop(['category_id', 'branch_id'], axis=1, inplace=True)

c_users_df['schedule_on'] = c_users_df['schedule_on'] + pd.Timedelta(days=1)
c_users_df['shifted_schedule_on'] = c_users_df['schedule_on']
# c_users_df['week_no'] = c_users_df['shifted_schedule_on'].dt.weekofyear
c_users_df['iso_calender'] = c_users_df['shifted_schedule_on'].dt.strftime('%G%V')
c_users_df["iso_calender"] = c_users_df["iso_calender"].astype(str).astype(int)
weeks = 0
weeks = c_users_df.groupby(['iso_calender']).size().reset_index(name='counts')
weeks.drop('counts', axis=1, inplace=True)
weeks['iso_calender'] = weeks['iso_calender'].sort_values(ascending=True)
weeks['week_no'] = weeks.index + 1
c_users_df = c_users_df.merge(weeks)
c_users_df.drop('iso_calender', axis=1, inplace=True)
orders_df = c_users_df.copy()

generate_client_ids = lambda x: str(int(x['company_id'])) + '_C' if pd.isna(x['building_id']) else str(
    int(x['building_id'])) + '_B'
orders_df['client_id'] = orders_df.apply(generate_client_ids, axis=1)
logger.info('At stage # %d', 5)
# %%
## calculate users and subscriptions
unique_count = lambda x: len(set(x))
c_users_df = orders_df.groupby(['client_id', 'week_no']).agg({'user_id': unique_count, 'quantity': 'sum'})

c_users_df.columns = ['last_week_unique_user_count', 'last_week_company_total_orders']
c_users_df.reset_index(inplace=True)
logger.info('At stage # %d', 6)
# %%

restaurants_df_1 = pd.read_csv(gap('restaurants.csv'), usecols=['id'])
restaurants_df_1.rename(columns={'id': 'restaurant_id'}, inplace=True)
restaurants_df_2 = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])
restaurants_df_2.rename(columns={'merlin_3_type': 'type', 'merlin_3_category': 'category'}, inplace=True)
restaurants_df_cols = ['restaurant_id']
restaurants_df = restaurants_df_2.merge(restaurants_df_1, on=restaurants_df_cols, how='inner')
restaurants_df = restaurants_df.dropna()

companies_df = pd.read_csv(gap('companies.csv'), usecols=['id', 'area_id'])
companies_df.rename(columns={'id': 'client_id'}, inplace=True)
companies_df['client_id'] = companies_df['client_id'].astype(int).astype(str) + '_C'
buildings_df = pd.read_csv(gap('buildings.csv'), usecols=['id', 'area_id'])
buildings_df.rename(columns={'id': 'client_id'}, inplace=True)
buildings_df['client_id'] = buildings_df['client_id'].astype(int).astype(str) + '_B'
client_df = companies_df.append(buildings_df)
logger.info('At stage # %d', 7)
# %%

master_df = c2r_sched_vs_orders_df[['schedule_on', 'client_id', 'restaurant_id', 'week_no', 'orders']].reset_index(
    drop=True).copy()
master_df['week_no'] = master_df['week_no'].transform(lambda x: x - 1)
logger.info('At stage # %d', 8)
# %%

old_df = pd.read_csv(gapp('all_prepared_scedules_vs_orders_with_rating_Qcount.csv'), parse_dates=['schedule_on'])
old_df = remove_scheduleon_holidays(old_df)
old_df = old_df.loc[old_df['schedule_on'].between(start_date, end_date)]
old_df['schedule_on'] = old_df['schedule_on'] + pd.Timedelta(days=1)
old_df = old_df.sort_values('schedule_on')
# old_df['week_no'] = old_df['schedule_on'].dt.weekofyear
old_df['iso_calender'] = old_df['schedule_on'].dt.strftime('%G%V')
old_df["iso_calender"] = old_df["iso_calender"].astype(str).astype(int)
weeks = 0
weeks = old_df.groupby(['iso_calender']).size().reset_index(name='counts')
weeks.drop('counts', axis=1, inplace=True)
weeks['iso_calender'] = weeks['iso_calender'].sort_values(ascending=True)
weeks['week_no'] = weeks.index + 1
old_df = old_df.merge(weeks)
old_df.drop('iso_calender', axis=1, inplace=True)

previous_orders_df = old_df.groupby(['restaurant_id', 'client_id', 'week_no']).tail(1).copy()
previous_orders_df['previous_orders'] = previous_orders_df.groupby(['restaurant_id', 'client_id'])['quantity'].shift(1)
previous_orders_df['previous_orders'] = previous_orders_df['previous_orders'].fillna(0)
previous_orders_df = previous_orders_df[['restaurant_id', 'client_id', 'week_no', 'previous_orders']]
old_df = old_df.merge(previous_orders_df, on=['restaurant_id', 'client_id', 'week_no'], how='left')
old_df['previous_orders'] = old_df['previous_orders'].bfill()

old_df = old_df.fillna(0)

old_df.rename(columns={'quantity': 'orders'}, inplace=True)
old_data_columns = ['client_id', 'restaurant_id', 'schedule_on']
master_df = master_df.merge(old_df[old_data_columns + ['previous_orders']], on=old_data_columns, how='left')

logger.info('At stage # %d', 9)
# %%
## adding company status on master table
master_df = master_df.merge(c_all_stats_df, on=['client_id', 'week_no'], how='left', suffixes=(False, '_company'))
master_df = master_df.merge(r_all_stats_df, on=['restaurant_id', 'week_no'], how='left',
                            suffixes=(False, '_restaurant'))
master_df = master_df.merge(c2r_all_stats_df, on=['client_id', 'restaurant_id', 'week_no'], how='left',
                            suffixes=(False, '_c2r'))

logger.info('max week # %d', c_all_stats_df['week_no'].max())
logger.info('max week # %d', r_all_stats_df['week_no'].max())
logger.info('max week # %d', c2r_all_stats_df['week_no'].max())

master_df = master_df.merge(c_users_df, on=['client_id', 'week_no'], how='left')

master_df = master_df.merge(restaurants_df, on='restaurant_id', how='inner')
master_df = master_df.merge(client_df, on='client_id', how='left')

logger.info('At stage # %d', 10)

#%%
orders_history_df = pd.read_csv(gap('order_histories.csv'))
orders_history_df['created_at'] = pd.to_datetime(orders_history_df['created_at'])
orders_history_df['expected_time'] = pd.to_datetime(orders_history_df['expected_time'])
orders_history_df = orders_history_df[(orders_history_df['created_at'].dt.normalize()).between(start_date, end_date)]
orders_history_df = orders_history_df.query('status == 5')
orders_history_df = orders_history_df[['order_id', 'status', 'expected_time', 'created_at']]
orders_history_df.columns = ['order_id', 'oh_status', 'oh_expected_time', 'oh_created_at']
orders_history_df = orders_history_df.drop_duplicates(['order_id', 'oh_status','oh_expected_time'], keep='last')
orders_history_df.head()

#%%

order_deliveries_df = pd.read_csv(gap('order_deliveries.csv'))
order_deliveries_df['created_at'] = pd.to_datetime(order_deliveries_df['created_at'])
order_deliveries_df['order_delivered_at'] = pd.to_datetime(order_deliveries_df['order_delivered_at'])
order_deliveries_df['delivery_notification_sent_at'] = pd.to_datetime(order_deliveries_df['delivery_notification_sent_at'])
order_deliveries_df = order_deliveries_df[(order_deliveries_df['created_at'].dt.normalize()).between(start_date, end_date)]
order_deliveries_df = order_deliveries_df[['order_id', 'delivery_notification_sent_at', 'order_delivered_at', 'created_at']]
order_deliveries_df.columns = ['order_id', 'od_delivery_notification_sent_at', 'od_order_delivered_at', 'od_created_at']

#%%
orders_ontime_df = pd.read_csv(gap('orders.csv'), parse_dates=['schedule_on']).rename(columns={'id': 'order_id'})
orders_ontime_df = orders_ontime_df[orders_ontime_df['schedule_on'].between(start_date, end_date)]
orders_ontime_df = remove_scheduleon_holidays(orders_ontime_df)
orders_ontime_df = orders_ontime_df[orders_ontime_df['category_id'] == 2]
orders_ontime_df = orders_ontime_df[orders_ontime_df['deleted_at'].isna()]
orders_ontime_df = orders_ontime_df[orders_ontime_df['status'] == 5]
orders_ontime_df = orders_ontime_df[['order_id', 'quantity', 'category_id', 'branch_id', 'schedule_on', 'company_id', 'building_id']]
orders_ontime_df.columns = ['order_id', 'quantity', 'category_id', 'branch_id', 'schedule_on', 'company_id', 'building_id']

#%%
master_ontime_df = orders_ontime_df.merge(orders_history_df, on='order_id', how='left')
master_ontime_df = master_ontime_df.merge(order_deliveries_df, on='order_id', how='left')

#%%
master_ontime_df['order_delivered_at'] = master_ontime_df['od_order_delivered_at'].fillna(master_ontime_df['oh_created_at'])
master_ontime_df['expected_vs_deliver_diff'] = (master_ontime_df.eval('order_delivered_at - oh_expected_time')/np.timedelta64(1,'m'))

master_ontime_df['client_id'] = master_ontime_df['building_id'].fillna(master_ontime_df['company_id'])
master_ontime_df['client_id'] = master_ontime_df['client_id'].astype(int).astype(str)
master_ontime_df.loc[master_ontime_df['building_id'].isna(), 'client_id'] = master_ontime_df.loc[master_ontime_df['building_id'].isna(), 'client_id'].astype(str) + '_C'
master_ontime_df.loc[~master_ontime_df['building_id'].isna(), 'client_id'] = master_ontime_df.loc[~master_ontime_df['building_id'].isna(), 'client_id'].astype(str) + '_B'

#%%
branches_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id']).rename(columns={'id': 'branch_id'})
master_ontime_df = master_ontime_df.merge(branches_df, on='branch_id', how='left')

master_ontime_df.loc[master_ontime_df['expected_vs_deliver_diff'] < -120, 'expected_vs_deliver_diff'] = -120
master_ontime_df.loc[master_ontime_df['expected_vs_deliver_diff'] > 120, 'expected_vs_deliver_diff'] = 120

master_ontime_df['schedule_on_shift1'] = master_ontime_df['schedule_on'] + pd.Timedelta(days=1)
master_ontime_df['week'] = master_ontime_df['schedule_on_shift1'].dt.year
mask = (master_ontime_df['schedule_on_shift1'].dt.week > 45) & (master_ontime_df['schedule_on_shift1'].dt.month == 1)
master_ontime_df.loc[mask, 'week'] = master_ontime_df.loc[mask, 'week'] - 1
master_ontime_df['week'] = master_ontime_df['week'].astype(str) + '_' + \
                    master_ontime_df['schedule_on_shift1'].dt.week.astype(str).transform(lambda x: x.zfill(2))

#%%
c2r_on_time_df = master_ontime_df.groupby(['schedule_on', 'week', 'client_id', 'restaurant_id']) \
    [['expected_vs_deliver_diff']].mean().reset_index()
c2r_on_time_df = master_ontime_df.groupby(['week', 'client_id', 'restaurant_id']) \
    [['expected_vs_deliver_diff']].mean().reset_index()

#%%
restaurant_ontime_df = master_ontime_df.groupby(['week', 'restaurant_id']) \
    [['expected_vs_deliver_diff']].mean().reset_index()

client_ontime_df = master_ontime_df.groupby(['week', 'client_id'])[['expected_vs_deliver_diff']].mean().reset_index()

#%%
c2r_on_time_df = c2r_on_time_df.sort_values('week')
client_ontime_df = client_ontime_df.sort_values('week')
restaurant_ontime_df = restaurant_ontime_df.sort_values('week')

c2r_on_time_df = c2r_on_time_df.query(' week > "2020_44"')
client_ontime_df = client_ontime_df.query(' week > "2020_44"')
restaurant_ontime_df = restaurant_ontime_df.query(' week > "2020_44"')
#%%
c2r_on_time_df_max = c2r_on_time_df.drop_duplicates(['client_id', 'restaurant_id'], keep='last')
client_ontime_df_max = client_ontime_df.drop_duplicates(['client_id'], keep='last')
restaurant_ontime_df_max = restaurant_ontime_df.drop_duplicates(['restaurant_id'], keep='last')

c2r_on_time_df_max.to_pickle(gapt('c2r_on_time_df_max.pkl'))
client_ontime_df_max.to_pickle(gapt('client_ontime_df_max.pkl'))
restaurant_ontime_df_max.to_pickle(gapt('restaurant_ontime_df_max.pkl'))
#%%
c2r_on_time_df['c2r_expected_vs_deliver_diff_shift1'] = c2r_on_time_df.groupby(['client_id', 'restaurant_id']) \
                ['expected_vs_deliver_diff'].shift()
client_ontime_df['c_expected_vs_deliver_diff_shift1'] = client_ontime_df.groupby(['client_id']) \
                ['expected_vs_deliver_diff'].shift()
restaurant_ontime_df['r_expected_vs_deliver_diff_shift1'] = restaurant_ontime_df.groupby(['restaurant_id']) \
                ['expected_vs_deliver_diff'].shift()

#%%
master_df['schedule_on_shift1'] = master_df['schedule_on'] + pd.Timedelta(days=0)
# master_df['week'] = master_df['schedule_on_shift1'].dt.year.astype(str) + '_' + master_df['schedule_on_shift1'].dt.week.astype(str).transform(lambda x: x.zfill(2))

master_df['week'] = master_df['schedule_on_shift1'].dt.year
mask = (master_df['schedule_on_shift1'].dt.week > 45) & (master_df['schedule_on_shift1'].dt.month == 1)
master_df.loc[mask, 'week'] = master_df.loc[mask, 'week'] - 1
master_df['week'] = master_df['week'].astype(str) + '_' + \
                    master_df['schedule_on_shift1'].dt.week.astype(str).transform(lambda x: x.zfill(2))

c2r_on_time_df_temp = c2r_on_time_df[['week', 'client_id', 'restaurant_id'] + ['c2r_expected_vs_deliver_diff_shift1']]
master_df = master_df.merge(c2r_on_time_df_temp, on=['week', 'client_id', 'restaurant_id'], how='left')

client_ontime_df_temp = client_ontime_df[['week', 'client_id' ] + ['c_expected_vs_deliver_diff_shift1']]
master_df = master_df.merge(client_ontime_df_temp, on=['week', 'client_id' ], how='left')

restaurant_ontime_df_temp = restaurant_ontime_df[['week',  'restaurant_id'] + ['r_expected_vs_deliver_diff_shift1']]
master_df = master_df.merge(restaurant_ontime_df_temp, on=['week',  'restaurant_id'], how='left')

master_df = master_df.merge(restaurant_ontime_df_max.rename(columns={'week':'max_week'}), on='restaurant_id',  how='left')
mask = (master_df['week'] > master_df['max_week']) & (master_df['r_expected_vs_deliver_diff_shift1'] != master_df['r_expected_vs_deliver_diff_shift1'])
master_df.loc[mask, 'r_expected_vs_deliver_diff_shift1'] = master_df.loc[mask, 'expected_vs_deliver_diff']
master_df.drop(columns=['max_week', 'expected_vs_deliver_diff'], inplace=True)

master_df = master_df.merge(client_ontime_df_max.rename(columns={'week':'max_week'}), on='client_id',  how='left')
mask = (master_df['week'] > master_df['max_week']) & (master_df['c_expected_vs_deliver_diff_shift1'] != master_df['c_expected_vs_deliver_diff_shift1'])
master_df.loc[mask, 'c_expected_vs_deliver_diff_shift1'] = master_df.loc[mask, 'expected_vs_deliver_diff']
master_df.drop(columns=['max_week', 'expected_vs_deliver_diff'], inplace=True)

master_df = master_df.merge(c2r_on_time_df_max.rename(columns={'week':'max_week'}), on=['client_id', 'restaurant_id'],  how='left')
mask = (master_df['week'] > master_df['max_week']) & (master_df['c2r_expected_vs_deliver_diff_shift1'] != master_df['c2r_expected_vs_deliver_diff_shift1'])
master_df.loc[mask, 'c2r_expected_vs_deliver_diff_shift1'] = master_df.loc[mask, 'expected_vs_deliver_diff']
master_df.drop(columns=['max_week', 'expected_vs_deliver_diff'], inplace=True)

ontime_columns = ['c2r_expected_vs_deliver_diff_shift1', 'c_expected_vs_deliver_diff_shift1', 'r_expected_vs_deliver_diff_shift1']
master_df[ontime_columns] = master_df[ontime_columns].fillna(0)

for i in ['schedule_on_shift1', 'week']:
    if i in master_df:
        master_df.drop(columns=[i], inplace=True)
# %%
# c_all_stats_df.to_csv('c_all_stats_df.csv', index=False)

c_all_stats_df_max = c_all_stats_df.groupby('client_id').apply(
    lambda x: x.nlargest(1, 'week_no', keep='first')).reset_index(drop=True)
c_all_stats_df_max.to_pickle(gapt('c_all_stats_df_max.pkl'))

r_all_stats_df_max = r_all_stats_df.groupby('restaurant_id').apply(
    lambda x: x.nlargest(1, 'week_no', keep='first')).reset_index(drop=True)
r_all_stats_df_max.to_pickle(gapt('r_all_stats_df_max.pkl'))

c2r_df_max = c2r_all_stats_df.groupby(['client_id', 'restaurant_id']).apply(
    lambda x: x.nlargest(1, 'week_no', keep='first')).reset_index(drop=True)
c2r_df_max.to_pickle(gapt('c2r_df_max.pkl'))

c_users_df_max = c_users_df[c_users_df['week_no'] == c_users_df['week_no'].max()].reset_index(drop=True)

c_users_df_max.to_pickle(gapt('c_users_df_max.pkl'))

restaurants_df.to_pickle(gapt('restaurants_df.pkl'))

client_df.to_pickle(gapt('client_df.pkl'))

# %%

master_df['schedule_on'] = master_df['schedule_on'] + pd.Timedelta(days=-1)
master_df['day_of_week'] = master_df['schedule_on'].dt.dayofweek
logger.info('At stage # %d', 11)
# %%

temp = master_df.pop('orders')
master_df['orders'] = temp
logger.info('At stage # %d', 12)

# %%
master_df.to_csv(gapp('temporal_c2r_QCount.csv'), index=False)

logger.info("--- %s Minutes ---" % ((time.time() - start_time) / 60))