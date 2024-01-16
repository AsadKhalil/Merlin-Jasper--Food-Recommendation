import pandas as pd
import os
from pathlib import Path
import json
import numpy as np
import argparse

pd.options.display.width = 0
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

#%%
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)

args = parser.parse_args()
data_base_path = args.data_dir #"D:\\office_work\\UAE_Analysis\\ds_active_work_space_ksa\\data"
# data_base_path = data_dir
raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)

#%%
gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
gapo = lambda x: os.path.join(processed_base_data_path, 'output', x)
#%%
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#%%

pred_df = pd.read_csv(gapp('ksa_ots.csv'), usecols=['branch_id', 'company_id', 'group_id', 'schedule_on', 'ots', 'restaurant_id']).rename(columns={'ots': 'orders'})
schedule_pairs_df = pd.read_csv(os.path.join(TARGET_OUTPUT_DIR, 'schedule_pairs.csv'))

restaurants_df = pd.read_csv(gap('restaurants.csv'), usecols=['id', 'name']).rename(columns={'id': 'restaurant_id', 'name': 'restaurant_name'})
branches_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'name', 'capacity', 'restaurant_id']).rename(columns={'id': 'branch_id', 'name': 'branch_name'})
groups_df = pd.read_csv(gap('groups.csv'), usecols=['id', 'per_day_restaurants_merlin3', 'name']).rename(columns={'id': 'group_id', 'name': 'group_name'})
resto_type_df = pd.read_csv(gap('restaurant_types.csv'), usecols=['id', 'name']).rename(columns={'id': 'merlin_3_type', 'name': 'merlin_3_type_name'})
resto_cat_df = pd.read_csv(gap('restaurant_categories.csv'), usecols=['id', 'name']).rename(columns={'id': 'merlin_3_category', 'name': 'merlin_3_category_name'})
resto_det_df = pd.read_csv(gap('restaurant_details.csv'), usecols=['restaurant_id', 'merlin_3_type', 'merlin_3_category'])

#%%
m_resto_df = branches_df.merge(restaurants_df, on='restaurant_id', how='left')
m_resto_df = m_resto_df.merge(resto_det_df, on='restaurant_id', how='left')
m_resto_df = m_resto_df.merge(resto_type_df, on='merlin_3_type', how='left')
m_resto_df = m_resto_df.merge(resto_cat_df, on='merlin_3_category', how='left')

#%%
resto_master_df = schedule_pairs_df.merge(m_resto_df, on=['branch_id', 'restaurant_id'], how='left')

#%%

cluster_df = pd.read_csv(gapp('sample_cluster_area.csv'), encoding = "ISO-8859-1")
cluster_df = cluster_df[['company_id', 'clustor_id', 'area_name', 'area_id', 'latitude', 'longitude']]
n_clustors = cluster_df[['area_id', 'clustor_id']].drop_duplicates().groupby('area_id').count()['clustor_id'].to_dict()
clustor_area_ids = list(n_clustors.keys())


companies_df = (pd.read_csv(gap('companies.csv'),
                            usecols=['id', 'area_id', 'status', 'is_active_virtual', 'group_id', 'latitude', 'longitude', 'deleted_at'])
                            .rename(columns={'id': 'company_id'}))
companies_df = companies_df[companies_df['deleted_at'].isna()].drop(columns=['deleted_at'])
companies_df = companies_df.dropna()
# companies_df = companies_df[companies_df['area_id'].isin(clustor_area_ids)]
max_group_in_area_df = (companies_df[companies_df['area_id'].isin(clustor_area_ids)].groupby('area_id')
                        .apply(lambda x: x.groupby('group_id').count().sort_values( 'company_id',ascending=False).index[0])
                        .reset_index(name='max_group'))
companies_df = companies_df.merge(max_group_in_area_df, on='area_id', how='left')
companies_df['max_group'] = companies_df['max_group'].fillna(companies_df['group_id'])
companies_df['group_id'] = companies_df['max_group']

companies_df = companies_df.drop(columns=['max_group'])

companies_df['group_id'] = companies_df['group_id'].astype(int).astype(str)

from sklearn.neighbors import KNeighborsClassifier
neigh = {}
for i in clustor_area_ids:
    temp_cluster_df = cluster_df[cluster_df['area_id'] == i]
    neigh[i] = KNeighborsClassifier(n_neighbors=1)
    neigh[i].fit(temp_cluster_df[['latitude', 'longitude']], temp_cluster_df['clustor_id'])

master_cluster_df = companies_df.merge(cluster_df[['company_id', 'clustor_id']], on='company_id', how='left')
mask = ((master_cluster_df['status'] == 1) & (master_cluster_df['is_active_virtual'] == 0))
master_cluster_df = master_cluster_df[mask]
master_cluster_df['missing_clusters'] = master_cluster_df['clustor_id']
mask = master_cluster_df['area_id'].isin(clustor_area_ids)
master_cluster_df.loc[mask, 'missing_clusters'] = master_cluster_df[mask].apply(lambda x: neigh[x['area_id']].predict([x[['latitude', 'longitude']].values])[0], axis=1)
master_cluster_df['clustor_id'] = master_cluster_df['clustor_id'].fillna(master_cluster_df['missing_clusters'])
master_cluster_df.drop(columns=['missing_clusters'], inplace=True)

mask = ~master_cluster_df['area_id'].isin(clustor_area_ids)
master_cluster_df.loc[mask, 'clustor_id'] = 0

master_cluster_df['group_id'] = master_cluster_df['group_id'] + '_' + master_cluster_df['clustor_id'].astype(int).astype(str)

mask = master_cluster_df['status'] == 2
# ha_companies_df = master_cluster_df[mask]
master_cluster_df = master_cluster_df[~mask]
master_cluster_df = master_cluster_df[['company_id', 'group_id', 'area_id']]

# ha_companies_df =

ss_pred_df = pred_df[['company_id', 'schedule_on', 'branch_id', 'restaurant_id', 'orders']]
master_cluster_df = master_cluster_df.merge(ss_pred_df, on='company_id', how='left')
master_cluster_df['orders'] = master_cluster_df['orders'].fillna(0)
master_cluster_df = master_cluster_df.groupby(['group_id', 'area_id', 'schedule_on', 'branch_id', 'restaurant_id'])['orders'].sum().reset_index()
#%%
resto_master_df = resto_master_df.merge(master_cluster_df, on=['group_id', 'schedule_on', 'branch_id', 'restaurant_id'], how='left')

groups_df = pd.read_csv(gap('groups.csv'), usecols=['id', 'name']).rename(columns={'id': 'actual_group_id', 'name': 'group_name'})

resto_master_df['actual_group_id'] = resto_master_df['group_id'].transform(lambda x: int(x.split('_')[0]))
resto_master_df['cluster_id'] = resto_master_df['group_id'].transform(lambda x: str(int(x.split('_')[1])))

resto_master_df = resto_master_df.merge(groups_df, on=['actual_group_id'], how='left')
resto_master_df['group_name'] = resto_master_df['group_name'] + '_' + resto_master_df['cluster_id']

resto_master_df['group_id'] = resto_master_df['group_id'].transform(lambda x: x.split('_')[0])
resto_master_df['group_id'] = resto_master_df['group_id'] + '0000' + resto_master_df['cluster_id']
resto_master_df['group_id'] = resto_master_df['group_id'].astype(int)
resto_master_df.drop(columns=['actual_group_id', 'cluster_id'], inplace=True)

resto_master_df['schedule_on'] = pd.to_datetime(resto_master_df['schedule_on'])
#%%
dd = resto_master_df
dd['schedules'] = dd['schedule_on'].dt.day_name()
# dd['group_name'] = 'group_name'
branch_trans_columns = ['branch_name', 'branch_id', 'capacity', 'restaurant_id', 'restaurant_name', 'schedules', 'group_id',
                        'group_name', 'schedule_status', 'orders']

branch_transform_df = dd[branch_trans_columns]
branch_transform_df.head()

#%%
branch_transform_df['branch_id'] = branch_transform_df['branch_id'].astype(str)
branch_transform_df['restaurant_id'] = branch_transform_df['restaurant_id'].astype(str)

cols = ['branch_name', 'branch_id', 'capacity', 'restaurant_id', 'restaurant_name']
lll = []
g = branch_transform_df.groupby(cols)
for (branch_name, branch_id, capacity, restaurant_id, restaurant_name), df in branch_transform_df.groupby(cols):
    mdict = {'branch_name' : branch_name,
             'branch_id': branch_id,
             'capacity': capacity,
             'restaurant_id': restaurant_id,
             'restaurant_name': restaurant_name,
             'schedules': {}}
    day_schedules = mdict['schedules']

    df.drop(columns=cols, inplace=True)
    for day_name, df2 in df.groupby('schedules'):
        df2.drop(columns=['schedules'], inplace=True)
        day_schedules[day_name] = eval(df2.to_json(orient='records'))

    lll.append(mdict)

with open(gapo('transformed_branches_data.json'), 'w') as json_file:
    json.dump(lll, json_file, indent=2, cls=NpEncoder)

#%%
menus_df = pd.read_csv(gap('menus.csv'), usecols=['id', 'name', 'restaurant_id', 'category_id', 'status', 'merlin_status']).rename(
                                            columns={'id': 'menu_id', 'name': 'menus_name'})
menus_df = menus_df.query('status == 1 and category_id == 2 and merlin_status == 1')
menus_df.drop(columns=['status', 'category_id', 'merlin_status'], inplace=True)
menus_df = menus_df.drop_duplicates('restaurant_id', keep='last')

menus_meal_df = pd.read_csv(gap('menus_meals.csv'), usecols=['menu_id', 'meal_id'])
meals_df = pd.read_csv(gap('meals.csv'), usecols=['id', 'name']).rename(columns={'id': 'meal_id', 'name': 'meal_name'})

menu_schedule_df = menus_df.merge(menus_meal_df, on='menu_id', how='left')
menu_schedule_df = menu_schedule_df.merge(meals_df, on='meal_id', how='left')
resto_master_df.head()
#%%

resto_master_df['branch_id'] = resto_master_df['branch_id'].astype(str)
resto_master_df['schedule_status'] = resto_master_df['schedule_status'].astype(str)

resto_meals = {}

for (restaurant_id, menus_name), df in menu_schedule_df.groupby(['restaurant_id', 'menus_name']):

    df2 = df[['meal_name']].rename(columns = {'meal_name': 'name'})
    json_meals = eval(df2.to_json(orient='records'))

    mdict = {'menu' : {
                    'name': menus_name,
                    'meal': json_meals}
             }
    resto_meals[restaurant_id] = mdict

resto_master_df['per_day_resto_allowed'] = 3

resto_master_df['schedules'] = resto_master_df['schedule_on'].dt.day_name()
resto_master_df['capacity'] = resto_master_df['capacity'].fillna(-1)
ll4 = []
cols = ['group_id', 'group_name', 'per_day_resto_allowed']
for (group_id, group_name, per_day_resto_allowed), df in resto_master_df.groupby(cols):

    mrow = {'group_id': group_id, 'group_name': group_name, 'per_day_resto_allowed': per_day_resto_allowed}
    mschedule = {'days': {}}
    mdays = mschedule['days']
    for row in df.itertuples():
        tt = mdays.setdefault(row.schedules, {})

        tt[row.merlin_3_type_name] = {
            'branch_name': row.branch_name,
            'branch_id': row.branch_id,
            'capacity': row.capacity,
            'schedule_status': row.schedule_status,
            'restaurant_category': row.merlin_3_category_name,
            'orders': row.orders,
            'menu': resto_meals[row.restaurant_id]['menu']}


    mrow['schedules'] = mschedule
    ll4.append(mrow)

with open(gapo('transformed_data.json'), 'w') as json_file:
    json.dump(ll4, json_file, indent=2, cls=NpEncoder)
