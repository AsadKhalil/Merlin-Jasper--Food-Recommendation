
from enum import Enum
import pandas as pd

class RestaurantTypes(Enum):
    New_Healthy = 10
    Grilled = 11
    Italian = 12
    Pan_Asian = 13
    Other = 9
    Burgers_Sandwiches = 14

    def __str__(self):
        return self.name


class RestaurantCategories(Enum):
    Warm_Bowls = 25
    Cold_Salads = 26
    Arabic_Grills = 1
    Mediterranean_Sandwiches_wraps = 21
    Pasta = 27
    Pizza = 22
    International = 17
    Thai = 24
    Chinese = 15
    Japanese = 19
    Korean = 20
    Indian = 8
    Mexican = 9
    Sushi = 11
    Mixed_Bag = 30
    Sandwiches = 28
    Burgers = 29

    def __str__(self):
        return self.name

def abstract_function(x):
    print('Please implement this function on main script')
    return x

gap = gapd = gapp = abstract_function

def drop_data_log(df, mask, drop_data_file_name):
    df[mask].to_csv(gapd(drop_data_file_name), index=False)


def get_resto_ids_having_active_menus():
    usecols = ['restaurant_id', 'category_id', 'status', 'merlin_status']
    menus_df = pd.read_csv(gap('menus.csv'), usecols=usecols, dtype={'category_id': 'int8', 'status': 'int8', 'merlin_status':'int8'})

    valid_mask = ((menus_df['status'] == 1) & (menus_df['merlin_status'] == 1) & (menus_df['category_id'] == 2))
    drop_data_log(menus_df, ~valid_mask, 'dropped_menus_wo_status1_or_merlinstatus1.csv')

    menus_df = menus_df[valid_mask]
    return menus_df['restaurant_id'].unique()


def get_active_branches_areas():
    branch_areas_df = pd.read_csv(gap('branch_area.csv'), usecols=['branch_id', 'area_id', 'is_deliverable'])
    branch_areas_df = branch_areas_df[branch_areas_df['is_deliverable'] == 1]
    branch_areas_df = branch_areas_df.groupby('branch_id')['area_id'].apply(list).reset_index(name='areas')
    return branch_areas_df


def get_areas_data():
    areas_df = (pd.read_csv(gap('areas.csv'), usecols=['id', 'merlin_blackout_period', 'city_id'])
                .rename(columns={'id': 'area_id'}))
    return areas_df


def drop_test_records(df, drop_data_file_name):
    valid_mask = df['name'].str.lower().str.contains('test')
    drop_data_log(df, ~valid_mask, drop_data_file_name)
    return df[valid_mask]


def get_branches_data():
    branches_df = (pd.read_csv(gap('branches.csv'),
                               usecols=['id', 'name', 'delivery_capacity', 'daily_dropoffs', 'status',
                                        'home_delivery_category_status'])
                   .rename(columns={'id': 'branch_id'}))
    drop_test_records(branches_df, 'dropped_test_branches.csv')

    valid_mask = (branches_df['status'].isin((1, 2)))
    drop_data_log(branches_df, ~valid_mask, 'dropped_inactive_branches.csv')
    branches_df = branches_df[valid_mask]

    valid_mask = ~(branches_df['home_delivery_category_status'] == 3)
    drop_data_log(branches_df, ~valid_mask, 'dropped_only_dinner_branches.csv')

    return branches_df[valid_mask].drop(['name'], axis=1)


def get_groups_data():
    groups = (pd.read_csv(gap('groups.csv'), usecols=['id', 'area_id', 'per_day_restaurants', 'is_perk_group'])
              .rename(columns={'id': 'group_id', 'is_perk_group': 'is_perk'}))
    return groups


def get_menu_schedules(start_date, end_date, client_type):
    if client_type == 'company':
        table_file_name = 'menu_schedules.csv'
        usecols = ['menu_id', 'company_id', 'schedule_on', 'branch_id', 'deleted_at']
        client_id_name = 'company_id'
    else:
        table_file_name = 'menu_schedules_building.csv'
        usecols = ['menu_id', 'building_id', 'schedule_on', 'branch_id', 'deleted_at']
        client_id_name = 'building_id'

    lunch_menus = (pd.read_csv(gap('menus.csv'), usecols=['id', 'category_id']).query('category_id == 2')['id'])

    schedules_df = pd.read_csv(gap(table_file_name), usecols=usecols, parse_dates=['schedule_on'])
    mask = ((schedules_df['schedule_on'].between(start_date, end_date)) & (schedules_df['deleted_at'].isna()) & (
        schedules_df['menu_id'].isin(lunch_menus)))
    schedules_df = schedules_df.loc[mask]
    schedules_df = schedules_df.drop_duplicates([client_id_name, 'schedule_on', 'branch_id'], keep='last')
    schedules_df = schedules_df.drop(columns=['deleted_at', 'menu_id'])

    return schedules_df


def get_company_groups():
    df = pd.read_csv(gap('companies.csv'), usecols=['id', 'group_id']).dropna()
    return df


def get_building_groups():
    df = (pd.read_csv(gap('buildings.csv'), usecols=['id', 'building_group_id'])
          .rename(columns={'building_group_id': 'group_id'}).dropna())
    return df


def get_group_level_menu_schedules(MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE):
    companies_schedule_df = get_menu_schedules(MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE, 'company')
    companies_schedule_df = companies_schedule_df.rename(columns={'company_id': 'id'})
    groups = get_company_groups()
    companies_schedule_df = companies_schedule_df.merge(groups, on='id', how='inner')

    building_schedule_df = get_menu_schedules(MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE, 'building')
    building_schedule_df = building_schedule_df.rename(columns={'building_id': 'id'})
    groups = get_building_groups()
    building_schedule_df = building_schedule_df.merge(groups, on='id', how='inner')

    schedules_df = pd.concat([companies_schedule_df, building_schedule_df])
    schedules_df = schedules_df.sort_values('schedule_on')
    schedules_df = schedules_df[['schedule_on', 'branch_id', 'group_id']].drop_duplicates()
    return schedules_df


def compute_blackouts(master_df):
    # Add a column for "days_diff" as a variable
    default_date = master_df['p_date'].min() + pd.Timedelta(days=-90)
    master_df['schedule_on'] = master_df['schedule_on'].fillna(default_date)
    master_df['days_diff'] = master_df['p_date'] - master_df['schedule_on']
    master_df['days_diff'] = master_df['days_diff'].dt.days

    # Create a flag column for "is_not_blackout"
    master_df['is_not_blackout'] = 0
    master_df.loc[master_df['merlin_blackout_period'] <= master_df['days_diff'], 'is_not_blackout'] = 1
    return master_df


def get_restaurant_data():
    df = pd.read_csv(gap('restaurants.csv'),
                     usecols=['id', 'name', 'status', 'category_status', 'type', 'category', 'is_premium']).rename(
        columns={'id': 'restaurant_id'})
    drop_test_records(df, 'dropped_test_restaurants.csv')

    valid_mask = df['status'].isin([1, 2])
    drop_data_log(df, ~valid_mask, 'dropped_invalid_status_restaurants.csv')
    df = df.loc[valid_mask]

    valid_mask = df['category_status'].isin([1, 3])
    drop_data_log(df, ~valid_mask, 'dropped_invalid_cat_status_restaurants.csv')
    df = df.loc[valid_mask]

    return df


def drop_invalid_resto_types(master_df):
    master_df = master_df[master_df['type'].isin([i.value for i in RestaurantTypes])]
    return master_df


def drop_invalid_resto_categories(master_df):
    master_df = master_df[master_df['category'].isin([i.value for i in RestaurantCategories])].copy()
    return master_df


def fix_perday_resto_to_default(master_df):
    master_df.loc[master_df['per_day_restaurants'] > 5, 'per_day_restaurants'] = 5
    master_df.loc[master_df['per_day_restaurants'] < 3, 'per_day_restaurants'] = 3
    return master_df


def set_default_flags(master_df):
    master_df['is_not_processes'] = True
    master_df['scheduled'] = False
    master_df['perday_resto'] = master_df['per_day_restaurants']
    return master_df

def menu_schedule_history(df):
    df = (df.groupby(['group_id', 'branch_id'])['schedule_on'].apply(list)
          .reset_index().rename(columns={'schedule_on': 'schedule_history'}))
    df['c2r_count'] = df['schedule_history'].transform(len)
    df['c2r_count'] = df['c2r_count'].fillna(0)
    return df

def attach_schedules_info(master_df, MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE):
    schedules_df = get_group_level_menu_schedules(MENU_SCHEDULE_START_DATE, MENU_SCHEDULE_END_DATE)

    sch_history = menu_schedule_history(schedules_df)
    master_df = master_df.merge(sch_history, on=['group_id', 'branch_id'], how='left')

    schedules_df = schedules_df[['group_id', 'branch_id', 'schedule_on']].drop_duplicates(
        subset=['group_id', 'branch_id'], keep='last')
    master_df = master_df.merge(schedules_df, on=['group_id', 'branch_id'], how='left')

    return master_df


def assign_week_numbers(df, datetime_col_name = 'p_date'):
    df['week_no'] = df[datetime_col_name].dt.weekofyear
    df['y_week_no'] = df[datetime_col_name].dt.year

    mask = ((df['week_no'] >= 51) & (df[datetime_col_name].dt.month == 1))
    df.loc[mask, 'y_week_no'] = df.loc[mask, 'y_week_no'] - 1
    df['y_week_no'] = df['y_week_no'] * 100
    df['y_week_no'] = df['y_week_no'] + df['week_no']
    return df