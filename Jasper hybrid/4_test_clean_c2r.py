import argparse
import logging
from datetime import datetime
import os, sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import kurtosis, skew

#%%

# python argstest.py --training_start_date "2020-11-01"  --data_dir "/temp"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)
parser.add_argument("--prediction_start_date", help="Start date of orders prediction from.", required=True)
parser.add_argument("--force_start_date", help="Actual start date from which order prepare script start. This "
                                                     "will override the --training_start_date parameter.")
parser.add_argument("--run_id", help="Script run_id helps in tracing and grouping logs in complete pipeline.", required=True)

args = parser.parse_args()
if args.force_start_date is not None:
    force_start_date = args.force_start_date
    force_start_date = datetime.strptime(force_start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    start_date = force_start_date
else:
    prediction_start_date = args.prediction_start_date
    prediction_start_date = datetime.strptime(prediction_start_date, '%Y-%m-%d')
    prediction_start_date = prediction_start_date + relativedelta(months=-3, days=-1)
    weekday = prediction_start_date.weekday() + 1
    week_day_diff = weekday % 7
    start_date = prediction_start_date + relativedelta(days=(week_day_diff * -1))
    start_date = start_date.strftime('%Y-%m-%d')
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
logger.debug('--args.prediction_start_date = %s', args.prediction_start_date)
logger.debug('--start_date = %s', start_date)
logger.debug('--data_dir = %s', data_base_path)
#%%

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
#%%

raw_base_data_path = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)

gap = lambda x: os.path.join(raw_base_data_path, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)
#%%
df = pd.read_csv(gapp('temporal_c2r_QCount.csv'))

#%%
logger.info('%s', df.shape)
logger.info('%s', df.drop_duplicates(['client_id', 'restaurant_id', 'schedule_on']).shape)
logger.info('%s', df.drop_duplicates().shape)
df[df.duplicated(subset=['client_id', 'restaurant_id', 'schedule_on'])].head(50)

#%%

df['schedule_on'] = pd.to_datetime(df['schedule_on'])
df = df.loc[df['schedule_on'] >= start_date].copy()

drop_columns_list=['rating_amaxFalse', 'rating_meanFalse', 'rating_medianFalse',
       'rating_aminFalse', 'rating_stdFalse', 'exp_orders_std',
       'exp_rating_kurtosisFalse', 'exp_rating_skewFalse', 'rating_skewFalse',
       'rating_kurtosisFalse', 'orders_std', 'rating_skew_restaurant',
       'rating_kurtosis_restaurant', 'level_2']
df.drop(drop_columns_list, axis=1, inplace=True)

df.dropna(subset=['restaurant_type', 'restaurant_category'], inplace=True)

kurtosis_columns = [i for i in list(df.columns) if 'kurtosis' in i]

df[kurtosis_columns] = df[kurtosis_columns].fillna(value=-3)

df['relative_prograssinve_schedule_rank'] = df['relative_prograssinve_schedule_rank'].fillna(value=1)

first_time_columns = ['is_first_time_c2r', 'is_first_time_company_schedule', 'is_first_time_restaurant_schedule']
df[first_time_columns] = df[first_time_columns].fillna(value=True)

df = df.fillna(0)

df.to_csv(gapp('temporal_c2r_QCount_clean.csv'), index=False)
logger.info('Data cleaning done')

