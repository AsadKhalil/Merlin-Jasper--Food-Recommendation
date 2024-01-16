import os, sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
import pandas as pd
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--prediction_end_date", help="End date of orders prediction from.", required=True)
parser.add_argument("--force_start_date", help="Actual start date from which order prepare script start. This "
                                                     "will override the --training_start_date parameter.")
parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.", required=True)

args = parser.parse_args()
prediction_start_date = None
train_start_date = test_start_date = None
train_end_date = test_end_date = None
prediction_end_date = args.prediction_end_date
if args.force_start_date is not None:
    force_start_date = args.force_start_date
    force_start_date = datetime.strptime(force_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    print(force_start_date)
    start_date = force_start_date
else:
    prediction_start_date = args.prediction_start_date
    prediction_start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
    test_start_date = prediction_start_date + relativedelta(days=-7)
    train_end_date = test_start_date + relativedelta(days=-1)
    weekday = test_start_date.weekday() + 1
    week_day_diff = weekday % 7
    test_start_date = test_start_date + relativedelta(days = (week_day_diff * -1))
    start_date = test_start_date.strftime('%Y-%m-%d')
    test_end_date = prediction_start_date + relativedelta(days=-1)
    print(start_date)

# end_date = datetime.strptime(args.prediction_end_date, '%Y-%m-%d')
# end_date = (prediction_start_date + relativedelta(days=-1)).strftime('%Y-%m-%d')

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
logger.debug('train_end_date = %s', train_end_date)
logger.debug('test_start_date = %s', test_start_date)
logger.debug('test_end_date = %s', test_end_date)
logger.debug('--data_dir = %s', data_base_path)
logger.debug('--run_id = %s', run_id)

raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
model_path = os.path.join(data_base_path, '..', 'models')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)
Path(model_path).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
gapm = lambda x: os.path.join(model_path, x)
start_time = time.time()
# assert False
#%%
def metrics_calculate(test_df):
    test_df = test_df[['Y', 'predictions']].copy()

    test_df['r_predictions'] = test_df['predictions'].transform(lambda x: 0 if x <= 0 else round(x))

    test_df['diff'] = test_df['Y'] - test_df['r_predictions']

    test_df['diff'] = test_df['diff'].transform(abs)

    diff_records = test_df['diff'].transform(lambda x: 0 if x <= 0 else 1).dropna().sum()

    diff_adj1_orders = test_df['diff'].transform(lambda x: 0 if x <= 1 else x).dropna().sum()
    diff_orders = test_df['diff'].transform(lambda x: 0 if x <= 0 else x).dropna().sum()

    diff_adj1_records = test_df['diff'].transform(lambda x: 0 if x <= 1 else 1).dropna().sum()
    # diff_records = diff.transform(lambda x: 0 if x <= 0 else 1).dropna().sum()

    records_size = len(test_df['Y'])

    mean_abs_error = mean_absolute_error(test_df['Y'], test_df['r_predictions'])
    mse = mean_squared_error(test_df['Y'], test_df['r_predictions'])
    rmse = sqrt(mse)
    r2 = r2_score(test_df['Y'], test_df['r_predictions'])
    mar = {'mean_abs_error': mean_abs_error, 'mse': mse, 'rmse': rmse, 'r2': r2}
    mar['y'] = test_df['Y'].sum()
    mar['y_hat'] = test_df['predictions'].sum()
    mar['overall_prediction_acc'] = min(mar['y_hat'], mar['y']) / max(mar['y_hat'], mar['y'])
    mar['diff_orders'] = diff_orders
    mar['diff_adj1_orders'] = diff_adj1_orders
    mar['diff_records'] = diff_records
    mar['diff_adj1_records'] = diff_adj1_records
    mar['acc'] = ((records_size - diff_records) / records_size)
    mar['acc_adj1'] = ((records_size - diff_adj1_records) / records_size)

    mar['acc_orders'] = ((test_df['Y'].sum() - diff_orders) / test_df['Y'].sum())
    mar['acc_adj1_orders'] = ((test_df['Y'].sum() - diff_adj1_orders) / test_df['Y'].sum())
    mar['#records'] = records_size
    mar['f1'] = f1_score(test_df['Y'], test_df['r_predictions'], average='macro')

    mar['actual_r2'] = r2_score(test_df['Y'], test_df['predictions'])

    return mar

#%%
df = pd.read_csv(gapp('temporal_c2r_QCount_clean.csv'))
df['schedule_on'] = pd.to_datetime(df['schedule_on'])
df.rename(columns={'restaurant_category': 'category', 'restaurant_type':'type'}, inplace=True)

df = df[df['category'].isin([1,2,8,14,17,18,21,22,23,26,27,28])]
df = df[df['type'].isin([1,5,6,7,9])]

df['client_type'] = df['client_id'].transform(lambda x: 0 if 'C' in x else 1)
logger.info('input dataframe size : %s', df.shape)
logger.info('At stage # %d', 1)
#%%
temp_type = df.pop('type')
temp_category = df.pop('category')
temp_area_id = df.pop('area_id')
temp_day_of_week = df.pop('day_of_week')
temp_orders = df.pop('orders')
temp_client_type = df.pop('client_type')

df['type'] = temp_type
df['category'] = temp_category
df['area_id'] = temp_area_id
df['day_of_week'] = temp_day_of_week
df['client_type'] = temp_client_type
df['orders'] = temp_orders

#%%
master_df = df.copy()
# data_window = ['2020-05-01', '2020-10-15']
# master_df = master_df.loc[master_df['schedule_on'].between(*data_window)]
# master_df = master_df[master_df['schedule_on'] != '2020-07-30']
# master_df = master_df[master_df['schedule_on'] != '2020-08-02']
# master_df = master_df[master_df['schedule_on'] != '2020-08-23']
# master_df = master_df[master_df['schedule_on'] != '2020-08-24']
logger.info('At stage # %d', 2)
#%%
drop = [ 'rating_amax_restaurant', 'exp_orders_median_restaurant', 'orders_medianFalse', 'exp_orders_amaxFalse', 'orders_kurtosis', 'exp_orders_skewFalse',
        'orders_amin', 'orders_kurtosis_restaurant', 'exp_orders_stdFalse', 'orders_mean_restaurant', 'exp_rating_stdFalse', 'orders_aminFalse', 'orders_skewFalse',
        'exp_orders_kurtosis_restaurant', 'exp_rating_kurtosis_restaurant', 'exp_rating_amax_restaurant',
        'orders_skew', 'orders_amin_restaurant', 'orders_std_restaurant', 'orders_amaxFalse' ,'orders_skew_restaurant',
        'exp_orders_medianFalse','exp_orders_skew', 'rating_amin_restaurant', 'exp_orders_kurtosis', 'exp_orders_kurtosisFalse',
        'orders_kurtosisFalse', 'orders_amax_restaurant', 'exp_orders_std_restaurant', 'exp_rating_amaxFalse', 'exp_orders_aminFalse', 'exp_orders_amin', 'exp_rating_aminFalse', 'exp_orders_skew_restaurant', 'exp_rating_skew_restaurant',
       'exp_rating_meanFalse',
 'exp_rating_medianFalse',
 'rating_mean_restaurant',
 'rating_median_restaurant',
 'exp_rating_mean_restaurant',
 'exp_rating_median_restaurant',
 'orders_mean',
 'orders_median',
 'exp_orders_median', 'area_id']
master_df.drop(columns=drop, inplace=True)
logger.info('Data selection complete')
logger.info('At stage # %d', 3)
#%%
master_columns = master_df.columns

train_data = master_df[master_df['schedule_on'] <= train_end_date]
test_data = master_df[master_df['schedule_on'].between(test_start_date, test_end_date)]

logger.info('Training data size : %s', train_data.shape)
logger.info('Validation data size : %s', test_data.shape)
logger.info('At stage # %d', 4)
#%%
identifier_columns = ['schedule_on', 'client_id', 'restaurant_id']
catagorical_columns = ['is_first_time_company_schedule', 'is_first_time_restaurant_schedule', 'is_first_time_c2r', 'category', 'day_of_week', 'type', 'client_type']
numarical_columns = [i for i in train_data.columns if i not in (catagorical_columns + identifier_columns + ['orders'])]

#%%

numeric_transformer = Pipeline(steps=[
    ('MinMax', MinMaxScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='error', drop='first'))])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numarical_columns),
        ('cat', categorical_transformer, catagorical_columns)])

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
model_obj = XGBRegressor(n_estimators = 150, objective='reg:squarederror',learning_rate=0.1)
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('refressor', model_obj)])
logger.info('Model architecture complete')
#%%
model.fit(train_data[numarical_columns + catagorical_columns], train_data['orders'])

from sklearn.externals import joblib
joblib.dump(model, gapm('cv_model.pkl'), compress = 1)
logger.info('1st Model Training complete')
logger.info('At stage # %d', 5)
#%%
# assert False

def save_model_results(results, results_name):
    model_results = pd.Series(results)
    model_results.to_csv(gapm(results_name + '.csv'), index=True)

#%%
predictions = model.predict(train_data[numarical_columns + catagorical_columns]).reshape(-1,1)
test_df = pd.DataFrame(train_data['orders'].tolist(), columns=['Y'])
test_df['predictions'] = predictions

logger.info("XGB predictions train score")
xgb_train_matrics = metrics_calculate(test_df)
save_model_results(xgb_train_matrics, gapm('cv_train_data_results'))
logger.info('%s', xgb_train_matrics)
#%%
print(test_data['category'].value_counts())
test_data = test_data[test_data['category'].isin(set(train_data['category']))]
test_data = test_data[test_data['type'].isin(set(train_data['type']))]
test_df = pd.DataFrame(test_data['orders'].tolist(), columns=['Y'])
test_df['predictions'] = model.predict(test_data[numarical_columns + catagorical_columns]).reshape(-1,1)
logger.info("XGB predictions validation score")
xgb_test_matrics = metrics_calculate(test_df)
save_model_results(xgb_test_matrics, gapm('cv_validation_data_results'))
logger.info('%s', xgb_test_matrics)
logger.info('At stage # %d', 6)
#%%
import pandas as pd
df2 = pd.read_csv(gapp('sample_temporal_c2r_clean.csv'))
df2['schedule_on'] = pd.to_datetime(df2['schedule_on'])
master_columns = master_df.columns
master_columns = list(master_columns)
master_columns.remove('orders')
logger.info('Prediction data size: %s', df2.shape)
logger.info('At stage # %d', 7)
#%%
df2.loc[~df2['category'].isin(list(set(train_data['category']))), 'category'] = None
df2.loc[~df2['type'].isin(list(set(train_data['type']))), 'category'] = None
df2.dropna(inplace=True)
# df2['category'] = df2.groupby('type', sort=False)['category'].transform(lambda x: x.fillna(x.mode().iloc[0]))
df2['client_type'] = df2['client_id'].transform(lambda x: 0 if 'C' in x else 1)
logger.info('At stage # %d', 8)
#%%
c_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', model_obj)])
c_model.fit(master_df[numarical_columns + catagorical_columns], master_df['orders'])
joblib.dump(model, gapm('complete_train_model.pkl'), compress = 1)
logger.info('Training on complete data complete')
logger.info('At stage # %d', 9)
#%%
test_df = pd.DataFrame(master_df['orders'].tolist(), columns=['Y'])
test_df['predictions'] = model.predict(master_df[numarical_columns + catagorical_columns]).reshape(-1,1)
logger.info("XGB predictions full train score")
xgb_train_matrics = metrics_calculate(test_df)
save_model_results(xgb_train_matrics, gapm('complete_train_model_data_results'))
logger.info('%s', xgb_train_matrics)
#%%
df2 = df2[master_columns]
predictions = c_model.predict(df2[numarical_columns + catagorical_columns]).reshape(-1,1)
#%%
results = df2[['schedule_on', 'client_id', 'restaurant_id']]
results['predictions'] = predictions
results['orders'] = results['predictions'].transform(lambda x: 0 if x <= 0 else round(x))
logger.info('At stage # %d', 10)
#%%
companies_df = pd.read_csv(gap('companies.csv'), usecols=['id', 'area_id']).rename(columns={'id':'company_id'})
# buildings_df = pd.read_csv(gap('buildings.csv'), usecols=['id', 'area_id']).rename(columns={'id':'building_id'})

branches_df = pd.read_csv(gap('branches.csv'), usecols=['id', 'restaurant_id', 'status']).rename(columns={'id':'branch_id'})
branches_df = branches_df.query('status == 1 or status == 2')
branches_df = branches_df[['branch_id', 'restaurant_id']]
branches_areas_df = pd.read_csv(gap('branch_area.csv'), usecols=['branch_id', 'area_id'])

branches_df = branches_df.merge(branches_areas_df, on='branch_id', how='inner')
#%%
companies_df['company_id'] = companies_df['company_id'].astype(int).astype(str) + '_C'
# buildings_df['building_id'] = buildings_df['building_id'].astype(int).astype(str) + '_B'

# buildings_df.rename(columns={'building_id': 'client_id'}, inplace=True)
companies_df.rename(columns={'company_id': 'client_id'}, inplace=True)

client_areas_df = companies_df.copy()
logger.info('At stage # %d', 12)
#%%
results = results.merge(client_areas_df, on='client_id', how='left')
# results = results.merge(buildings_df, on='building_id', how='left')
logger.info('Prediction data before attaching branch_id : ,%s', results.shape)
results = results.merge(branches_df, on=['restaurant_id', 'area_id'], how='left')
logger.info('Prediction data after attaching branch_id : ,%s', results.shape)
results = results[['schedule_on', 'client_id', 'restaurant_id', 'branch_id', 'orders']].dropna()
results['schedule_on'] = results['schedule_on'] + timedelta(days=7)
# Loading comp_groups file to filter out companies for Jasper only
comp_groups = pd.read_csv(gapp('comp_groups.csv'))
results['company_id'] = results['client_id'].transform(lambda x: int(x.split('_')[0]) if 'C' == x.split('_')[1] else None)
results.drop(["client_id"], inplace=True, axis=1)
results = pd.merge(results, comp_groups[['company_id','Model']], on='company_id', how='inner')


results = results[['company_id', 'schedule_on', 'branch_id', 'restaurant_id', 'Model', 'orders']]
ksa_ots = pd.read_csv(gapp('ksa_ots.csv'))
ksa_ots = ksa_ots.groupby(['company_id','group_id']).size().reset_index(name='count')
ksa_ots.drop(["count"], inplace=True, axis=1)

#%%

results = pd.merge(results, ksa_ots[['company_id','group_id']], on='company_id', how='left')
results = results[['company_id', 'group_id', 'schedule_on', 'branch_id', 'restaurant_id', 'orders', 'Model']]
jasper_pred = results.copy()
jasper_pred = jasper_pred[['company_id', 'group_id', 'schedule_on', 'branch_id', 'restaurant_id', 'orders']]
jasper_pred.to_csv(gapp('ksa_jasper.csv'), index=False)

jasper_pred_comp = results.loc[results['Model'] == 'Jasper']
ots_pred_comp = pd.read_csv(gapp('ots_pred_comp.csv'))
ots_pred_comp.rename(columns={'ots':'orders'}, inplace=True)
final_results = pd.concat([ots_pred_comp, jasper_pred_comp], axis=0, sort=False)
final_results.to_csv(gapp('jasper_hybrid_predictions.csv'), index=False)

logger.info('#Done')
logger.info("--- %s Minutes ---" % ((time.time() - start_time) / 60))

#%%