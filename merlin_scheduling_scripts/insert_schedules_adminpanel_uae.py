import argparse
import os
import pandas as pd
from dataconfig import MONGO_DB_NAME, MONGO_READER_HOST, sql_engine, exc # Use MONGO_DB_NAME for local mongo dump
import json

TARGET_OUTPUT_DIR = '/home/ds_active_work_space/data/processed/v6/output'
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--INPUT_FILE", help="The scheduled results file in csv format.", default=os.path.join(TARGET_OUTPUT_DIR, 'schedule_pairs.csv'))
parser.add_argument("--OUTPUT_DROP_OFF_FILE", help="The path to store drop_off wise json file.", default=os.path.join(TARGET_OUTPUT_DIR, 'transformed_data.json'))
parser.add_argument("--OUTPUT_BRANCHES_FILE", help="The path to store branches wise json file.", default=os.path.join(TARGET_OUTPUT_DIR, 'transformed_branches_data.json'))
parser.add_argument("--MASTER_SCHEDULE_FILE", help="The path to get the master_scheduling_result.csv file.", default=os.path.join(TARGET_OUTPUT_DIR,  'master_scheduling_result.csv'))
parser.add_argument("--insert_company_schedules_adminpanel", help="Pass True if want to insert in adminpanel else default is False", default=False)
parser.add_argument("--insert_buildings_schedules_adminpanel", help="Pass True if want to insert in adminpanel else default is False", default=False)
parser.add_argument("--insert_schedules_dashbord", help="Pass True if want to insert in adminpanel else default is False", default=False)

args = parser.parse_args()
data_base_path = args.data_dir
raw_base_data_path = os.path.join(data_base_path, 'raw')

processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output')




gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)

companies_df = pd.read_csv(gap('companies.csv'), usecols=['id' , 'drop_off_point_id', 'area_id', 'status', 'is_active_virtual', 'group_id'])
menus_df = pd.read_csv(gap('menus.csv'), usecols=['id', 'name', 'restaurant_id', 'category_id', 'status', 'merlin_status'])
pred_df = pd.read_csv(gapp('clientId_sample_predictions.csv'), usecols=['branch_id', 'client_id', 'schedule_on', 'restaurant_id', 'orders'])

pred_df['schedule_on'] = pd.to_datetime(pred_df['schedule_on']) + pd.Timedelta(days=1)
pred_df['schedule_on'] = pred_df['schedule_on'].dt.strftime('%Y-%m-%d')

def local_mongo_dump(output_drop_off_file, output_branches_file):
    # # dump into local mongodb
    # # read transformed_data (drop_off_point_id)
    print("Dumping json data to local mongo ..")
    with open(output_drop_off_file) as j_file:
        data = json.load(j_file)

    drop_off_collec = MONGO_DB_NAME['transformed_data']
    drop_off_collec.remove()
    drop_off_collec.insert(data)

    # read transformed_data (drop_off_point_id)

    with open(output_branches_file) as j_file:
        bran_data = json.load(j_file)

    drop_off_collec = MONGO_DB_NAME['transformed_branches_data']
    drop_off_collec.remove()
    drop_off_collec.insert(bran_data)

def get_menu_schedules_df(input_file):
    print("Preparing menu schedules dataframe for companies ..")
    schedule_pairs_df = pd.read_csv(input_file, parse_dates=['schedule_on'])
    scheduled_comps_df = schedule_pairs_df[schedule_pairs_df['drop_off_point_id'].str.contains('CD')]
    scheduled_buildings_df = schedule_pairs_df[schedule_pairs_df['drop_off_point_id'].str.contains('B')]
    scheduled_comps_df['drop_off_point_id'] = scheduled_comps_df['drop_off_point_id'].str.replace('_CD', '')

    # Merge company_id
    comps_df = companies_df.astype({"drop_off_point_id": str})
    comps_df = comps_df[(comps_df['status']==1) & (comps_df['is_active_virtual']==0)].drop_duplicates()
    # Uncomment this to exclude the shared drop_off_point_id companies
    # comps_merged = scheduled_comps_df.merge(comps_df.drop_duplicates(subset=['drop_off_point_id']), on="drop_off_point_id", how="left").rename(columns={'id': 'company_id'})
    comps_merged = scheduled_comps_df.merge(comps_df, on="drop_off_point_id", how="left").rename(columns={'id': 'company_id'})

    # Merge menu_id
    curr_menus_df = menus_df[(menus_df['category_id']==2) & (menus_df['status']==1) & (menus_df['merlin_status']==1)].drop_duplicates('restaurant_id', keep='last')
    df_merged = comps_merged.merge(curr_menus_df, on="restaurant_id", how="left").rename(columns={'id': 'menu_id'})
    # df_merged = df_merged[(df_merged['category_id']==2) & (df_merged['status_y']==1) & (df_merged['merlin_status']==1)]

    # Merge orders
    df_merged['schedule_on'] = pd.to_datetime(df_merged['schedule_on'])
    pred_df['schedule_on'] = pd.to_datetime(pred_df['schedule_on'])
    order_merged_df = df_merged.merge(pred_df.drop_duplicates(subset=["branch_id", "restaurant_id", "schedule_on"]), on=["branch_id", "restaurant_id", "schedule_on"], how="left").rename(columns={'orders': 'ots'})

    comps_result_df = order_merged_df[['schedule_on','branch_id', 'company_id', 'menu_id', 'ots']]
    return comps_result_df

#%%
def get_menu_schedules_building_df(input_file):
    print("Preparing menu scheules dataframe for buildings ..")
    schedule_pairs_df = pd.read_csv(input_file, parse_dates=['schedule_on'])
    scheduled_buildings_df = schedule_pairs_df[schedule_pairs_df['drop_off_point_id'].str.contains('B')]

    scheduled_buildings_df['drop_off_point_id'] = scheduled_buildings_df['drop_off_point_id'].str.replace('_B', '')

    # Creating building_id
    scheduled_buildings_df = scheduled_buildings_df.rename(columns={'drop_off_point_id': 'building_id'})

    # Merge menu_id
    curr_menus_df = menus_df[(menus_df['category_id']==2) & (menus_df['status']==1) & (menus_df['merlin_status']==1)].drop_duplicates('restaurant_id', keep='last')
    df_merged = scheduled_buildings_df.merge(curr_menus_df, on="restaurant_id", how="left").rename(columns={'id': 'menu_id'})

    # Merge orders
    df_merged['schedule_on'] = pd.to_datetime(df_merged['schedule_on'])
    pred_df['schedule_on'] = pd.to_datetime(pred_df['schedule_on'])
    order_merged_df = df_merged.merge(pred_df.drop_duplicates(subset=["branch_id", "restaurant_id", "schedule_on"]), on=["branch_id", "restaurant_id", "schedule_on"], how="left").rename(columns={'orders': 'ots'})

    building_res_df = order_merged_df[['schedule_on','branch_id', 'building_id', 'menu_id', 'ots']]
    return building_res_df


def mysql_dump_companies(comps_df):
    print("Inserting into menu_schedules ..")
    # create a temp table in mysql and then run INSERT-IGNORE statement
    comps_df.to_sql(name='temp_menu_schedules', con=sql_engine, if_exists='replace', index=False)
    with sql_engine.begin() as cn:
        delete_empty = """DELETE FROM temp_menu_schedules WHERE menu_id is null;"""
        cn.execute(delete_empty)
        sql = """INSERT INTO menu_schedules (company_id, menu_id, branch_id, ots, schedule_on, created_at, updated_at)
                    SELECT t.company_id, t.menu_id, t.branch_id, t.ots, t.schedule_on, now() as created_at, now() as updated_at
                    FROM temp_menu_schedules t
                    WHERE NOT EXISTS 
                        (SELECT * FROM menu_schedules f
                        WHERE t.company_id = f.company_id
                        AND t.menu_id = f.menu_id
                        AND t.branch_id = f.branch_id
                        AND t.schedule_on = f.schedule_on)"""

        try:
            cn.execute(sql)

        except exc.IntegrityError as e:
            errorInfo = e.orig.args
            print(errorInfo[0])  # This will give you error code
            print(errorInfo[1])  # This will give you error message
            raise e
            pass



def mysql_dump_buildings(builds_df):
    print("Inserting into menu_schedules_building ..")
    # create a temp table in mysql and then run INSERT-IGNORE statement
    builds_df.to_sql(name='temp_menu_schedules_building', con=sql_engine, if_exists='replace', index=False)
    with sql_engine.begin() as cn:
        delete_empty = """DELETE FROM temp_menu_schedules_building WHERE menu_id is null;"""
        cn.execute(delete_empty)
        sql = """INSERT INTO menu_schedules_building (building_id, menu_id, branch_id, ots, schedule_on, created_at, updated_at)
                    SELECT t.building_id, t.menu_id, t.branch_id, t.ots, t.schedule_on, now() as created_at, now() as updated_at
                    FROM temp_menu_schedules_building t
                    WHERE NOT EXISTS 
                        (SELECT 1 FROM menu_schedules_building f
                        WHERE (t.building_id = f.building_id
                        OR  (t.building_id IS NULL AND f.building_id IS NULL))
                        AND (t.menu_id = f.menu_id
                        OR  (t.menu_id IS NULL AND f.menu_id IS NULL))
                        AND t.branch_id = f.branch_id
                        AND t.schedule_on = f.schedule_on)"""
        try:
            cn.execute(sql)

        except exc.IntegrityError as e:
            errorInfo = e.orig.args
            print(errorInfo[0])  # This will give you error code
            print(errorInfo[1])  # This will give you error message
            raise e
            pass


    # flushing temp tables
    with sql_engine.begin() as e_con:
        sql = """DROP TABLE IF EXISTS temp_menu_schedules, temp_menu_schedules_building"""
        e_con.execute(sql)

if __name__ == "__main__":


    #args = parser.parse_args()
    INPUT_FILE = args.INPUT_FILE
    OUTPUT_DROP_OFF_FILE = args.OUTPUT_DROP_OFF_FILE
    OUTPUT_BRANCHES_FILE = args.OUTPUT_BRANCHES_FILE
    MASTER_SCHEDULE_FILE = args.MASTER_SCHEDULE_FILE

    if args.insert_schedules_dashbord:
        local_mongo_dump(OUTPUT_DROP_OFF_FILE, OUTPUT_BRANCHES_FILE)
        print("insert_schedules_dashbord complete")

    if args.insert_company_schedules_adminpanel:
        comps_df = get_menu_schedules_df(INPUT_FILE)
        mysql_dump_companies(comps_df)
        print("insert_company_schedules_adminpanel complete")

    if args.insert_buildings_schedules_adminpanel:
        builds_df = get_menu_schedules_building_df(INPUT_FILE)
        mysql_dump_buildings(builds_df)
        print("insert_buildings_schedules_adminpanel complete")


