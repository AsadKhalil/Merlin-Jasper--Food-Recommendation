# %%
import timeit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import merlin_utils

start = timeit.default_timer()
import os
from pathlib import Path
from datetime import datetime
import argparse
from dateutil.relativedelta import relativedelta
from merlin_utils import *
from datetime import datetime, timedelta
pd.options.display.width = 0
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)



#%%


# %%
def _parse_arguments():
    """ Parse arguments provided by commend line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.",
                        required=False, default='2021-01-03')
    parser.add_argument("--data_dir", help="Data directory location.",
                        required=False, default='D:\\office_work\\jasper2_beta2\\ds_workspace\\data')
    parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.",
                        required=False, default='local')
    parser.add_argument("--prediction_file_name", help="Name of client to restaurant (C2R) prediction file.",
                        required=False, default='group_prediction_uae.csv')
    
    parser.add_argument(
        '-f',
        '--file',
        help='Path for input file. First line should contain number of lines to search in'
    )
    return parser.parse_args()

def check_dateformat(date, format='%Y-%m-%d'):
    return datetime.strptime(date, format).strftime('%Y-%m-%d')

args = _parse_arguments()

data_base_path = args.data_dir
START_DATE = check_dateformat(args.prediction_start_date)

prediction_file_name = args.prediction_file_name
MENU_SCHEDULE_WINDOWS_IN_MONTHS = -3

#%%

# %%
RAW_DATA_FILES_PATH = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'merlin_output')
TARGET_OUTPUT_LOG_DIR = os.path.join(processed_base_data_path, 'dropped_data/scheduling_dropped_data_dir')

Path(TARGET_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TARGET_OUTPUT_LOG_DIR).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(RAW_DATA_FILES_PATH, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
gapd = lambda x: os.path.join(TARGET_OUTPUT_LOG_DIR, x)

merlin_utils.gap = gap
merlin_utils.gapp = gapp
merlin_utils.gapd = gapd

#%%


print('wait')


def data_prep():
    pred_orders_df = pd.read_csv(gapp(prediction_file_name), parse_dates=['schedule_on'])
    NUMBER_OF_SCHEDULING_DAYS = pred_orders_df['schedule_on'].nunique()

    MENU_SCHEDULE_END_DATE = (pred_orders_df['schedule_on'].min() + pd.Timedelta(-1, unit='d'))
    MENU_SCHEDULE_START_DATE = MENU_SCHEDULE_END_DATE + relativedelta(months=MENU_SCHEDULE_WINDOWS_IN_MONTHS)

    active_menu_resto_list = get_resto_ids_having_active_menus()
    pred_orders_df = pred_orders_df[pred_orders_df['restaurant_id'].isin(active_menu_resto_list)]

    areas_df = get_areas_data()
    branch_areas_df = get_active_branches_areas()
    branches_df = get_branches_data()

    branches_df = branches_df.merge(branch_areas_df, on='branch_id', how='left')
    valid_mask = ~branches_df['areas'].isna()
    drop_data_log(branches_df, ~valid_mask, 'dropped_branches_wo_areas.cav')
    branches_df = branches_df[valid_mask]

    groups_df = get_groups_data()
    master_df = pred_orders_df.merge(groups_df, on='group_id', how='left')
    master_df = master_df.merge(areas_df, on=['area_id'], how='left')
    master_df = master_df.merge(branches_df[['branch_id', 'delivery_capacity']], on='branch_id', how='inner')
    master_df = master_df.rename(columns={'schedule_on': 'p_date'})

    master_df = attach_schedules_info(master_df, MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE)
    master_df['day'] = master_df['p_date'].dt.dayofweek
    master_df = assign_week_numbers(master_df)
    master_df = compute_blackouts(master_df)

    resto_df = get_restaurant_data()
    master_df = master_df.merge(resto_df, on='restaurant_id', how='inner')

    master_df = drop_invalid_resto_types(master_df)
    master_df = drop_invalid_resto_categories(master_df)

    master_df = fix_perday_resto_to_default(master_df)

    master_df['CR_pair'] = master_df['group_id'].astype(str) + '_' + master_df['restaurant_id'].astype(str)

    master_df['CMoD'] = master_df.groupby(['p_date', 'group_id'])['orders'].transform('max')
    master_df['BMoD'] = master_df.groupby(['p_date', 'branch_id'])['orders'].transform('max')

    master_df = set_default_flags(master_df)
    master_df = master_df.rename(columns={'schedule_on': 'menu_schedule_on'})
    return master_df

RC = RestaurantCategories
cuisine_table = [
    (RC.Warm_Bowls, None, None),
    (RC.Cold_Salads, None, None),
    (RC.Arabic_Grills, None, None),
    (RC.Mediterranean_Sandwiches_wraps, 3, 3),
    (RC.Pasta, None, None),
    (RC.Pizza, 1, 1),
    (RC.International, 1, 1),
    (RC.Thai, 3, 3),
    (RC.Chinese, 3, 3),
    (RC.Japanese, 2, 2),
    (RC.Korean, 2, 2),
    (RC.Indian, 3, 3),
    (RC.Mexican, 3, 3),
    (RC.Sushi, 2, 2),
    (RC.Mixed_Bag, 1, 1),
    (RC.Sandwiches, None, None),
    (RC.Burgers, 2, 2),
]
cuisine_table = pd.DataFrame(cuisine_table, columns=['categories', 3, 4])
cuisine_table['id'] = cuisine_table['categories'].apply(lambda x: getattr(RestaurantCategories, str(x)).value)
cuisine_table = cuisine_table.set_index('id')


RT = RestaurantTypes
resto_type = [
    (RT.New_Healthy, 1, 2),
    (RT.Grilled, None, 1),
    (RT.Italian, None, None),
    (RT.Pan_Asian, None, None),
    (RT.Other, None, 2),
    (RT.Burgers_Sandwiches, None, None),
]
resto_type_table = pd.DataFrame(resto_type, columns=['types', 3, 4])
resto_type_table['id'] = resto_type_table['types'].apply(lambda x: getattr(RestaurantTypes, str(x)).value)
resto_type_table = resto_type_table.set_index('id')



#%%


def remove_non_schedule_pairs(active_df, non_scheduled_list, mask, remove_mask, reason, iteration_no):
    non_scheduled_list.append(active_df.loc[remove_mask].assign(reason=reason).copy().assign(iteration_no=iteration_no))
    active_df = active_df.loc[mask]
    return active_df

active_pool_dict = {}

def attest_active_zones(p_date, branch_id, group_id, branches_pool, is_checking):
    day_branches_pool = branches_pool[branches_pool['day'] == p_date]
    branch = day_branches_pool[day_branches_pool['branch_id'] == branch_id].iloc[0]
    delivery_capacity = branch['delivery_capacity']

    if p_date not in active_pool_dict:
        active_pool_dict[p_date] = {}

    if branch_id not in active_pool_dict[p_date]:
        t_dict = active_pool_dict[p_date]
        t_dict[branch_id] = list()
        active_pool_dict[p_date] = t_dict

    capacity_status = len(active_pool_dict[p_date][branch_id]) >= delivery_capacity
    if capacity_status:
        return False, 'capacity_full'

    if is_checking:
        return ~capacity_status, 'checking'

    active_pool_dict[p_date][branch_id].append(group_id)
    capacity_status = len(active_pool_dict[p_date][branch_id]) < delivery_capacity

    return True, 'single' if capacity_status else 'all'
import numpy as np
def assign_weights(df):
    df['c2r_count'] = df['c2r_count'].fillna(0)
    df['is_new_resto'] = np.where(df['c2r_count'] == 0 , 1, 0)
    df['is_new_resto'] = np.where(((df['c2r_count'] == 1) & (df['days_diff'] >= 45)), 2, df['is_new_resto'])

    df['sch_weight'] = np.where(df['c2r_count'] == 0 , 200, 0)
    df['sch_weight'] = np.where(((df['c2r_count'] == 1) & (df['days_diff'] < 14)), 100, df['sch_weight'])
    df['sch_weight'] = np.where(((df['c2r_count'] == 1) & (df['days_diff'] >= 45)), 50, df['sch_weight'])

    return df


def parse_yweek_no(datetime, days_shift=0):
    week_no = (datetime).weekofyear
    y_week_no = (datetime).year

    if ((week_no >= 51) and (datetime.month == 1)):
        y_week_no = y_week_no - 1

    y_week_no = y_week_no * 100
    y_week_no = y_week_no + week_no
    return y_week_no
	
	
#%%

def prepare_relevance_data(week_start_no, week_end_no):
    all_prep_df = pd.read_parquet(gapp('all_prepared_clean.parquet.gzip'))
    # all_prep_df = assign_week_numbers(all_prep_df['schedule_on'], datetime_col_name='schedule_on')

    # %%
    def filter_active_clients(df, week_start_no, week_end_no, min_active_orders=5):
        active_df = df[df['y_week_no'].between(week_start_no, week_end_no)]
        active_df = active_df.groupby('client_id')['delivered_orders'].sum()

        print(f"Total clients count = {active_df.shape[0]}")
        active_df = active_df[active_df > min_active_orders]
        print(f"Total active clients count on > {min_active_orders} orders filter = {active_df.shape[0]}")

        active_df = active_df.reset_index()
        return df.loc[df['client_id'].isin(active_df['client_id'])]

    all_prep_df = filter_active_clients(all_prep_df, week_start_no, week_end_no)

    act_all_prep_df = all_prep_df.groupby(['schedule_on', 'group_id', 'restaurant_id', 'branch_id', 'y_week_no']).agg({
        'rating': 'mean', 'delivered_orders': 'sum', 'undelivered_orders': 'sum', 'client_id': 'count'}).reset_index()

    act_all_prep_df = act_all_prep_df.rename(columns={'client_id': 'group_size', 'group_id': 'client_id'})
    return act_all_prep_df

def get_last_n_weeks(datetime, n_weeks):
    end_date = datetime
    start_date = end_date + timedelta(weeks = n_weeks)
    weeks = pd.date_range(start_date, end_date, freq='W').to_frame(name='schedule_on')
    weeks = assign_week_numbers(weeks, datetime_col_name='schedule_on')

    return weeks['y_week_no']

def relevance_base_type(df,  week_lags=5):
    p_date = df.iloc[0]['p_date']
    week_no = parse_yweek_no(p_date)
    weeks_list = get_last_n_weeks(p_date, n_weeks=-5)
    week_start_no = min(weeks_list)
    week_end_no = max(weeks_list)

    tmp = prepare_relevance_data(week_start_no, week_end_no)

    tmp = tmp[tmp['y_week_no'].between(week_start_no, week_end_no)]

    tmp['day_total_orders'] = tmp.groupby(['schedule_on', 'client_id'])['delivered_orders'].transform('sum')
    tmp['day_resto_pct'] = tmp['delivered_orders'] / tmp['day_total_orders']
    tmp2 = tmp.groupby(['client_id', 'restaurant_id'])['day_resto_pct'].mean().reset_index().rename(columns={'day_resto_pct': 'resto_pct'})

    avsp : pd.DataFrame = df.copy()
    avsp = avsp.rename(columns={'group_id': 'client_id'})
    # avsp = avsp[avsp['week_no'] == week_no]
    avsp = avsp.merge(tmp, on=['restaurant_id', 'client_id'], how='left')
    avsp = avsp.merge(tmp2, on=['restaurant_id', 'client_id'], how='left')
    # avsp['perday'] = avsp.groupby(['schedule_on', 'client_id'])['branch_id'].transform('count')
    avsp['resto_pct'] = avsp['resto_pct'].fillna(1 / avsp['per_day_restaurants'])
    #avsp['pct_sum'] = avsp.groupby(['schedule_on', 'client_id'])['resto_pct'].transform('sum')
    #avsp['adj_resto_pct'] = avsp['resto_pct'] / avsp['pct_sum']
    #avsp['daily_orders'] = avsp.groupby(['schedule_on', 'client_id'])['delivered_orders'].transform('sum')

    # avsp['predictions'] = avsp['adj_resto_pct'] * avsp['daily_orders']
    avsp = avsp[['client_id', 'restaurant_id', 'resto_pct', 'per_day_restaurants']].drop_duplicates()
    return avsp
	
#%%

def assign_groups(x):
    perday_resto = x['per_day_restaurants'].iloc[0]
    perday_resto = int(perday_resto)
    labels = np.arange(0, 1, 1/perday_resto) + (1/perday_resto)

    x['rank'] = x['resto_pct'].rank(method = 'first')
    x['rank_bin'] = pd.qcut(x['rank'], perday_resto, labels=labels)

    return x

def sort_scheduling_order(df):
    sort_columns=['sch_weight', 'orders', 'rank_bin','rank']
    return df.sort_values(sort_columns, ascending=False)

#%%
master_df = data_prep()

print("DE complete")

branches_pool = master_df[['day', 'branch_id', 'delivery_capacity']].drop_duplicates().copy()

master_df = assign_weights(master_df)

print('pause')

#%%

relevance_table = relevance_base_type(master_df[['restaurant_id', 'group_id', 'p_date', 'per_day_restaurants']], week_lags=5)
relevance_table = relevance_table.groupby('client_id').apply(lambda x: assign_groups(x))

if 'group_id' in master_df:
    master_df = master_df.rename(columns={'group_id': 'client_id'})
master_df = master_df.merge(relevance_table, on=['client_id', 'restaurant_id', 'per_day_restaurants'], how='left')

#%%

