
#%%
import argparse
import logging
from datetime import datetime
import os, sys
from pathlib import Path
import re
import time
import pandas as pd
from scipy.stats import kurtosis, skew


#%%


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--prediction_end_date", help="End date of orders prediction from.", required=True)
parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.", required=True)

args = parser.parse_args()

end_date = datetime.strptime(args.prediction_end_date, '%Y-%m-%d').strftime('%Y-%m-%d')
start_date = datetime.strptime(args.prediction_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
base_data_path = args.data_dir
run_id = args.run_id

# prediction_start_date = "2021-02-14" 
# prediction_end_date = "2021-02-20"
# data_dir = "/home/munchon/Documents/Munir/Documents/merlin_py/data" 
# run_id = 'local'

log_dir_path = os.path.join(base_data_path, '..', 'logs')
Path(log_dir_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig( level=logging.DEBUG,
    format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt = '%Y-%m-%d:%H:%M:%S',
    handlers = [ logging.FileHandler(Path(log_dir_path).joinpath(run_id+'.log')), logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger(__name__)

logger.debug('--start_date = %s', start_date)
logger.debug('--end_date = %s', end_date)
logger.debug('--data_dir = %s', base_data_path)
logger.debug('--run_id = %s', run_id)




#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
from pandas.tseries import offsets
from datetime import timedelta

#%%
# start_date = '2020-10-11'
# end_date = '2020-10-15'

data_for_clients = 'ALL'
start_time = time.time()
# %%

raw_base_data_path = os.path.join(base_data_path, 'raw')
processed_base_data_path = os.path.join(base_data_path, 'processed', 'v6')

TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output/dropped_data_prediction_dir')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)
Path(TARGET_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
gapt = lambda x: os.path.join(processed_base_data_path, 'pickles', x)

c_all_stats_df_max = pd.read_pickle(gapt('c_all_stats_df_max.pkl'))
r_all_stats_df_max = pd.read_pickle(gapt('r_all_stats_df_max.pkl'))
c2r_df_max = pd.read_pickle(gapt('c2r_df_max.pkl'))
c_users_df_max = pd.read_pickle(gapt('c_users_df_max.pkl'))
# restaurants_df = pd.read_pickle('pickles/restaurants_df.pkl')
# client_df = pd.read_pickle('pickles/client_df.pkl')
logger.info('At stage # %d', 1)
#%%
c2r_df_max['relative_prograssinve_schedule_rank'] = c2r_df_max['relative_prograssinve_schedule_rank'].fillna(2)
r_all_stats_df_max['is_first_time_restaurant_schedule'] = False

#%%
dates = pd.date_range(start=start_date, end=end_date, freq='1D').to_frame(index=False)[0].tolist()

#%%

companies_df = pd.read_csv(gap('companies.csv'), usecols=['id', 'status', 'area_id', 'name'])

#%%
# Dropping companies with status !=1 or have test in the names
dropped_comps_df = companies_df[(companies_df['status'] != 1) | (companies_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE))]
dropped_comps_df.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_inactive_or_test_companies.csv'), index=False)
#%%
companies_df = companies_df[companies_df['status'] == 1].copy()
companies_df = companies_df[~companies_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)].drop('name',axis=1)
companies_df['id'] = companies_df['id'].astype(str) + '_C'
companies_df.rename(columns={'id': 'client_id'}, inplace=True)

buildings_df = pd.read_csv(gap('buildings.csv'), usecols=['id', 'status', 'area_id', 'name'])

#%%
# Dropping buildings with status !=1 or have test in the names
dropped_builds_df = buildings_df[(buildings_df['status'] != 1) | (buildings_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE))]
dropped_builds_df.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_inactive_or_test_buildings.csv'), index=False)

#%%

dropped_comps_df=dropped_comps_df.rename(columns={'id': 'company_id', 'name': 'company_name'})
dropped_builds_df=dropped_builds_df.rename(columns={'id': 'building_id', 'name': 'building_name'})

dropped_builds_df['reason']='test/inactive building'
dropped_comps_df['reason']='test/inactive company'

dropped_clients_combined = pd.concat([dropped_comps_df, dropped_builds_df], ignore_index=True, sort=False)
#%%
buildings_df = buildings_df[buildings_df['status'] == 1].copy()
buildings_df = buildings_df[~buildings_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)].drop('name',axis=1)
buildings_df['id'] = buildings_df['id'].astype(str) + '_B'
buildings_df.rename(columns={'id': 'client_id'}, inplace=True)

client_df = None

if data_for_clients == 'C':
    client_df = companies_df
elif data_for_clients == 'B':
    client_df = buildings_df
else:
    client_df = companies_df.append(buildings_df)
#%%
unique_client_list = client_df['client_id'].unique().tolist()
restaurants_df_1 = pd.read_csv(gap('restaurants.csv'), usecols=['id', 'status', 'name'])
restaurants_df_1.rename(columns={'id': 'restaurant_id'}, inplace=True)
restaurants_df_2 = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])
restaurants_df_2.rename(columns={'merlin_3_type': 'type', 'merlin_3_category': 'category'}, inplace=True)
#%%
# Dropping nan restaurants
dropped_nan_restos = restaurants_df_2[restaurants_df_2.isnull().any(axis=1)]
# Dropping invalid type restaurants
dropped_invalid_type_restos = restaurants_df_2[~restaurants_df_2['type'].isin([4, 5, 6, 7, 8, 9])].dropna()
# Dropping invalid category restaurants
dropped_invalid_category_restos = restaurants_df_2[~restaurants_df_2['category'].isin([1, 8, 9, 11] + list(range(14, 26)))].dropna()

#%%
dropped_nan_restos['reason']='empty resto type/category'
dropped_invalid_type_restos['reason']='invalid resto type'
dropped_invalid_category_restos['reason']='invalid resto category'

dropped_restos_combined = pd.concat([dropped_nan_restos, dropped_invalid_type_restos, dropped_invalid_category_restos], ignore_index=True, sort=False)
dropped_restos_combined = dropped_restos_combined.merge(restaurants_df_1, on='restaurant_id', how='left')
dropped_restos_combined = dropped_restos_combined.rename(columns={'name': 'restaurant_name'})
#%%
restaurants_df_2 = restaurants_df_2.dropna()
restaurants_df_cols = ['restaurant_id']
restaurants_df_2 = restaurants_df_2[restaurants_df_2['type'].isin([4, 5, 6, 7, 8, 9])]
restaurants_df_2 = restaurants_df_2[restaurants_df_2['category'].isin([1, 8, 9, 11] + list(range(14, 26)))]
restaurants_df = restaurants_df_1.merge(restaurants_df_2, on=restaurants_df_cols, how='inner')
#%%
# Dropping restos without status 1 or 2
dropped_inactive_restos = restaurants_df[~restaurants_df['status'].isin([1,2])]
# Dropping restos with test names
dropped_test_restos = restaurants_df[restaurants_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)]

#%%
dropped_inactive_restos = dropped_inactive_restos.rename(columns={'name': 'restaurant_name'})
dropped_test_restos = dropped_test_restos.rename(columns={'name': 'restaurant_name'})
dropped_restos_combined = pd.concat([dropped_test_restos, dropped_restos_combined], ignore_index=True, sort=False)
dropped_restos_combined = dropped_restos_combined.rename(columns={'status': 'resto_status', 'type': 'resto_type', 'category': 'resto_category'})
#%%
# Logging dropped restos
dropped_nan_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_nan_restos.csv'), index=False)
dropped_invalid_type_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_invalid_type_restos.csv'), index=False)
dropped_invalid_category_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_invalid_category_restos.csv'), index=False)
dropped_inactive_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_inactive_restos.csv'), index=False)
dropped_test_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_test_restos.csv'), index=False)
#%%
restaurants_df = restaurants_df.dropna()
restaurants_df = restaurants_df[restaurants_df['status'].isin([1,2])]
restaurants_df = restaurants_df[~restaurants_df['name'].str.contains('test', na=False, regex=True, flags=re.IGNORECASE)].drop('name', axis=1)
#%%
restaurants_df = restaurants_df.dropna()
unique_restaurant_list = restaurants_df['restaurant_id'].unique().tolist()
logger.info('At stage # %d', 2)

#%%
import itertools

C2R_product = itertools.product(dates, unique_client_list, unique_restaurant_list)
C2R_product = list(C2R_product)

master_df = pd.DataFrame(C2R_product, columns=['schedule_on', 'client_id', 'restaurant_id'])
#%%
mask = (pd.Series(master_df['schedule_on'].dt.dayofweek).isin([4, 5])) & (master_df['client_id'].str.endswith('_C'))
master_df = master_df[~mask].copy()
logger.info('At stage # %d', 3)
#%%
brances_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id', 'status']).rename(columns={'id': 'branch_id'})
#%%
# Dropping branches without status 1 or 2
dropped_inactive_branches = brances_df[~brances_df['status'].isin([1, 2])]
dropped_inactive_branches.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_inactive_branches.csv'), index=False)

#%%
dropped_inactive_branches = dropped_inactive_branches.rename(columns={'status': 'branch_status'})
dropped_inactive_branches = dropped_inactive_branches.merge(restaurants_df_1, on ='restaurant_id', how='left')
dropped_inactive_branches = dropped_inactive_branches.rename(columns={'name': 'resto_name'})
dropped_inactive_branches['reason']= 'inactive branch'
del dropped_inactive_branches['status']
#%%
dropped_restos_combined = dropped_restos_combined.merge(dropped_inactive_branches, on='restaurant_id', how='outer')
#%%
# dropped_restos_combined = pd.concat([dropped_inactive_branches, dropped_restos_combined], ignore_index=True, sort=False)

#%%
dropped_restos_combined.restaurant_name.fillna(dropped_restos_combined.resto_name, inplace=True)
del dropped_restos_combined['resto_name']
dropped_restos_combined = dropped_restos_combined.rename(columns={'reason_x': 'resto_reason', 'reason_y': 'branch_reason'})
#%%
brances_df = brances_df[brances_df['status'].isin([1, 2])]

branch_areas_df = pd.read_csv(gap('branch_area.csv'), usecols=['branch_id', 'area_id'])
#%%
# Dropping branches without branch_areas
dropped_branches_without_areas = brances_df.merge(branch_areas_df, on='branch_id', how='inner', indicator=True)
dropped_branches_without_areas = dropped_branches_without_areas[dropped_branches_without_areas['_merge'] != 'both']
dropped_branches_without_areas.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_branches_without_areas.csv'), index=False)
#%%
# merge with combined resto df
dropped_branches_without_areas['reason']='invalid branch area'
dropped_restos_combined = dropped_restos_combined.merge(pd.notnull(dropped_branches_without_areas.branch_id), on='branch_id', how='outer')
#%%
brances_df = brances_df.merge(branch_areas_df, on='branch_id', how='inner')
#%%
# Dropping branches without valid restos
dropped_branches_wo_valid_restos = brances_df.groupby('restaurant_id')['area_id'].apply(list).reset_index(name='areas')
temp_df = brances_df
# dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos[~dropped_branches_wo_valid_restos['restaurant_id'].isin(restaurants_df['restaurant_id'])].copy()
dropped_branches_wo_valid_restos = temp_df[~temp_df['restaurant_id'].isin(restaurants_df['restaurant_id'])].drop_duplicates()
dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.drop_duplicates(['branch_id','restaurant_id'],keep= 'last')
dropped_branches_wo_valid_restos.to_csv(gapp(TARGET_OUTPUT_DIR + '/dropped_branches_wo_valid_restos'), index=False)
#%%
# Dropping branches without valid restos
dropped_branches_wo_valid_restos['branch_reason']= 'branch dropped due to invalid resto'
dropped_branches_wo_valid_restos['branch_id'] = dropped_branches_wo_valid_restos.branch_id.astype(float)
dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.rename(columns={'status': 'branch_status'})

# change status to branch_status, add restaurant_name, type and category in dropped_branches_wo_valid_restos
dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.merge(restaurants_df_1, on='restaurant_id', how='left')
restaurants_details_df = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])

dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.merge(restaurants_details_df, on='restaurant_id', how='left')
dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.rename(columns={'status': 'branch_status', 'name': 'restaurant_name', 'status': 'resto_status', 'merlin_3_type': 'resto_type', 'merlin_3_category': 'resto_category'})
#%%
# re-indexing
# dropped_branches_wo_valid_restos = dropped_branches_wo_valid_restos.reset_index()
# dropped_restos_combined = dropped_restos_combined.reset_index()

dropped_restos_combined = pd.concat([dropped_restos_combined, dropped_branches_wo_valid_restos], ignore_index=True, sort=False)
#%%
brances_df = brances_df.groupby('restaurant_id')['area_id'].apply(list).reset_index(name='areas')

#%%

# Dropping branches without valid restos
dropped_branches_wo_valid_resto = brances_df[~brances_df['restaurant_id'].isin(restaurants_df['restaurant_id'])].copy()
#%%
del dropped_branches_wo_valid_resto['areas']
#%%
dropped_branches_wo_valid_resto['branch_reason']= 'invalid resto'

# add restaurant_name, type and category in dropped_branches_wo_valid_restos
dropped_branches_wo_valid_resto = dropped_branches_wo_valid_resto.merge(restaurants_df_1, on='restaurant_id', how='left')
restaurants_details_df = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])
dropped_branches_wo_valid_resto = dropped_branches_wo_valid_resto.rename(columns={'status': 'resto_status', 'name': 'restaurant_name', 'merlin_3_type': 'resto_type', 'merlin_3_category': 'resto_category'})

dropped_branches_wo_valid_resto = dropped_branches_wo_valid_resto.merge(restaurants_details_df, on='restaurant_id', how='left')
#%%
dropped_restos_combined = pd.concat([dropped_restos_combined, dropped_branches_wo_valid_resto], ignore_index=True, sort=False)

#%%
# Logging dropped data combined file

dropped_restos_combined.to_csv(gapp(TARGET_OUTPUT_DIR + '/developer_dropped_combined_resto_branch.csv'), index=False)
dropped_clients_combined.to_csv(gapp(TARGET_OUTPUT_DIR + '/developer_dropped_combined_clients.csv'), index=False)
#%%
brances_df = brances_df[brances_df['restaurant_id'].isin(restaurants_df['restaurant_id'])].copy()
logger.info('At stage # %d', 4)
#%%

master_df = master_df.merge(client_df[['client_id', 'area_id']], on='client_id', how='left')
#%%
master_df = master_df.merge(brances_df, on='restaurant_id', how='left')
master_df['area_id'] = master_df['area_id'].astype(int)

master_df = master_df.dropna()
master_df['is_deliver_in_area'] = master_df.apply(lambda x: x['area_id'] in x['areas'], axis=1)
master_df = master_df[master_df['is_deliver_in_area']].copy()
master_df.drop(columns=['area_id', 'areas', 'is_deliver_in_area'], inplace=True)

master_df['week_no'] = int(c_all_stats_df_max['week_no'].max())
logger.info('At stage # %d', 5)
#%%
c_all_stats_df_max.drop(columns=['week_no'], inplace=True)
r_all_stats_df_max.drop(columns=['week_no'], inplace=True)
c2r_df_max.drop(columns=['week_no'], inplace=True)
c_users_df_max.drop(columns=['week_no'], inplace=True)

master_df = master_df.merge(c_all_stats_df_max, on=['client_id'], how='left', suffixes=(False, '_company'))
master_df = master_df.merge(r_all_stats_df_max, on=['restaurant_id'], how='left', suffixes=(False, '_restaurant'))
master_df = master_df.merge(c2r_df_max, on=['client_id', 'restaurant_id'], how='left', suffixes=(False, '_c2r'))

# del (c_all_stats_df, r_all_stats_df, c2r_all_stats_df)

master_df = master_df.merge(c_users_df_max, on=['client_id'], how='left')

master_df = master_df.merge(restaurants_df[['restaurant_id', 'type', 'category']], on='restaurant_id', how='left')
master_df = master_df.merge(client_df, on='client_id', how='left')
logger.info('At stage # %d', 6)
#%%
# master_df['schedule_on'] = master_df['schedule_on'] + pd.Timedelta(days=-1)
master_df['day_of_week'] = master_df['schedule_on'].dt.dayofweek


#%%

old_df = pd.read_csv(gapp('all_prepared_scedules_vs_orders_with_rating_Qcount.csv'), parse_dates=['schedule_on'])
old_df = old_df[(old_df['schedule_on'] >= '2020-01-01') & (old_df['schedule_on'] <= end_date)]
old_df.drop(columns=['rating'], inplace=True)
# old_df.drop(columns=['rating'], inplace=True)
old_df['schedule_on'] = old_df['schedule_on'] + pd.Timedelta(days=1)
old_df = old_df.sort_values('schedule_on')
logger.info('At stage # %d', 7)
#%%
import time
s = time.time()
previous_orders_od_df = old_df.copy()
previous_orders_od_df['schedule_on'] = previous_orders_od_df['schedule_on'] + pd.Timedelta(days=-1)
previous_orders_od_df = previous_orders_od_df[previous_orders_od_df['schedule_on'] < start_date]
previous_orders_od_df = previous_orders_od_df.sort_values('schedule_on', ascending=True)
previous_orders_od_df = previous_orders_od_df.groupby(['client_id', 'restaurant_id']).tail(1).reset_index(drop=True)

previous_orders_od_df['week_no'] = master_df['week_no'].max()
previous_orders_od_df.rename(columns={'quantity': "previous_orders"}, inplace=True)
previous_orders_od_df.drop(columns=['schedule_on'], inplace=True)

master_df = master_df.merge(previous_orders_od_df, on=['client_id', 'restaurant_id', 'week_no'], how='left')

master_df['orders'] = -1
logger.info('At stage # %d', 8)
#%%

c2r_on_time_df_max = pd.read_pickle(gapt('c2r_on_time_df_max.pkl'))
client_ontime_df_max = pd.read_pickle(gapt('client_ontime_df_max.pkl'))
restaurant_ontime_df_max = pd.read_pickle(gapt('restaurant_ontime_df_max.pkl'))

c2r_on_time_df_max.rename(columns={'expected_vs_deliver_diff': 'c2r_expected_vs_deliver_diff_shift1'}, inplace=True)
client_ontime_df_max.rename(columns={'expected_vs_deliver_diff': 'c_expected_vs_deliver_diff_shift1'}, inplace=True)
restaurant_ontime_df_max.rename(columns={'expected_vs_deliver_diff': 'r_expected_vs_deliver_diff_shift1'}, inplace=True)
_=c2r_on_time_df_max.pop('week')
_=client_ontime_df_max.pop('week')
_=restaurant_ontime_df_max.pop('week')

#%%
master_df = master_df.merge(c2r_on_time_df_max, on=['client_id', 'restaurant_id'], how='left')
master_df = master_df.merge(client_ontime_df_max, on=['client_id'], how='left')
master_df = master_df.merge(restaurant_ontime_df_max, on=['restaurant_id'], how='left')


#%%
prepared_data = pd.read_csv(gapp('temporal_c2r_QCount_clean.csv'), parse_dates=['schedule_on'])
# prepared_data['schedule_on'] = pd.to_datetime(prepared_data['schedule_on'])
# prepared_data = prepared_data[prepared_data['week_no'] != max(prepared_data['week_no'])]
df = prepared_data.copy()

master_df = master_df[df.columns]


master_df[['orders_kurtosis', 'exp_orders_kurtosis']] = master_df[['orders_kurtosis', 'exp_orders_kurtosis']].fillna(-3)
master_df['is_first_time_restaurant_schedule'] = master_df['is_first_time_restaurant_schedule'].fillna(True)
master_df['is_first_time_company_schedule'] = master_df['is_first_time_company_schedule'].fillna(True)
master_df['is_first_time_c2r'] = master_df['is_first_time_c2r'].fillna(True)
master_df['relative_prograssinve_schedule_rank'] = master_df['relative_prograssinve_schedule_rank'].fillna(1)

ontime_columns = ['c2r_expected_vs_deliver_diff_shift1', 'c_expected_vs_deliver_diff_shift1', 'r_expected_vs_deliver_diff_shift1']
master_df[ontime_columns] = master_df[ontime_columns].fillna(0)

master_df = master_df.fillna(0)
logger.info('At stage # %d', 9)
#%%
temp_type = master_df.pop('type')
temp_category = master_df.pop('category')
temp_area_id = master_df.pop('area_id')
temp_day_of_week = master_df.pop('day_of_week')
temp_orders = master_df.pop('orders')

master_df['type'] = temp_type
master_df['category'] = temp_category
master_df['area_id'] = temp_area_id
master_df['day_of_week'] = temp_day_of_week
master_df['orders'] = temp_orders

master_df.to_csv(gapp('sample_temporal_c2r_clean.csv'), index=False)

logger.info(10)
logger.info("--- %s Minutes ---" % ((time.time() - start_time) / 60))
