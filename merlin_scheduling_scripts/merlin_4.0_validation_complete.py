#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import numpy as np


# In[2]:


data_base_path = 'C:\\Users\\LunchON\\Desktop\\validation'
RAW_DATA_FILES_PATH = os.path.join(data_base_path, 'raw')
processed_base_data_path = os.path.join(data_base_path, 'processed', 'v6')
TARGET_OUTPUT_DIR = os.path.join(processed_base_data_path, 'output')
Path(processed_base_data_path).mkdir(parents=True, exist_ok=True)
Path(TARGET_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TARGET_OUTPUT_LOG_DIR = os.path.join(processed_base_data_path, 'output/dropped_data_scheduling_dir')
Path(TARGET_OUTPUT_LOG_DIR).mkdir(parents=True, exist_ok=True)


# In[3]:


gap = lambda x: os.path.join(RAW_DATA_FILES_PATH, x)
gapp = lambda x: os.path.join(processed_base_data_path, x)


# In[4]:


data_prep_file =pd.read_csv("Master.csv")
merlin4_schedules_2 =pd.read_csv("merlin4_schedules_2.csv").rename(columns={'client_id': 'group_id'})


# In[5]:


merlin4_schedules_2 = merlin4_schedules_2[['p_date','group_id','branch_id','restaurant_id']]


# In[6]:


merlin4_schedules_2 = merlin4_schedules_2.merge(data_prep_file,how='inner',on=['p_date','group_id','branch_id','restaurant_id'])


# In[7]:


merlin4_schedules_2


# In[8]:


merlin4_schedules_2['schedule_status'] = 0


# In[9]:


merlin4_schedules_2.to_csv("validation_file.csv",index=False)


# In[10]:


TEST_REPORT =[]


# ### NEW RESTO RULE

# In[11]:


merlin4_schedules_2['new_resto'] = False
merlin4_schedules_2['partial_new_resto'] =False
merlin4_schedules_2.loc[(merlin4_schedules_2['days_diff']>90), 'new_resto'] = True
merlin4_schedules_2.loc[(merlin4_schedules_2['days_diff'].isna()), 'new_resto'] = True
merlin4_schedules_2.loc[ (merlin4_schedules_2['days_diff']>45) & (merlin4_schedules_2['days_diff']<90), 'partial_new_resto'] = True


# In[12]:


def checking_new_restos(scheduled_pairs,TEST_REPORT):
    #Checking New Resto 
    # Not Assigned more than one time in a day
    groups_have_new_restos_per_day=scheduled_pairs.groupby(['p_date','group_id'])['new_resto'].apply                                 (lambda x : x[x==True].count()).reset_index()
    groups_have_new_restos_per_day = groups_have_new_restos_per_day[groups_have_new_restos_per_day['new_resto']>1]
    no_of_groups_have_new_restos = groups_have_new_restos_per_day.shape[0]
    
    
    TEST_REPORT.append(('Comparative', 'New_Resto',
                        'is_any_new_resto_schedule', no_of_groups_have_new_restos!=0))
    TEST_REPORT.append(('Comparative', 'New_Resto',
                        '#Of_any_new_resto_schedule', no_of_groups_have_new_restos))
    TEST_REPORT.append(('Comparative', 'New_Resto',
                        'new_resto_schedule', '<<<no_of_groups_have_new_restos>>>'))
    
    groups_have_new_restos_per_day.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'no_of_groups_have_new_restos.csv')),index=False)

    
def checking_partial_new_restos(scheduled_pairs,TEST_REPORT):
    #Checking partial new restos 
    # Not assigned more than 2 in a week
    
    groups_have_partial_restos_per_week =scheduled_pairs.groupby(['group_id'])['partial_new_resto'].apply                                         (lambda x : x[x==True].count()).reset_index()
    groups_have_partial_restos_per_week =groups_have_partial_restos_per_week[groups_have_partial_restos_per_week['partial_new_resto']>2]
    
    no_groups_have_partial_restos_per_week =groups_have_partial_restos_per_week.shape[0]
    
    TEST_REPORT.append(('Comparative', 'partial_Resto',
                        'is_any_partial_resto_schedule_per_week', no_groups_have_partial_restos_per_week!=0))
    TEST_REPORT.append(('Comparative', 'partial_Resto',
                        '#is_any_partial_resto_schedule_per_week', no_groups_have_partial_restos_per_week))
    TEST_REPORT.append(('Comparative', 'partial_Resto',
                        'no_groups_have_partial_restos_per_week', '<<<no_of_groups_have_partial_restos>>>'))
    

    groups_have_partial_restos_per_week.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'no_of_groups_have_partial_restos.csv')),index=False)
    

def new_and_partial_resto_assign_together(scheduled_pairs,TEST_REPORT):
    
    both_resto_assign_together =scheduled_pairs.groupby(['p_date','group_id'])['new_resto','partial_new_resto'].sum().reset_index()
    both_resto_assign_together =both_resto_assign_together[ (both_resto_assign_together['new_resto']>0) &                                                            (both_resto_assign_together['partial_new_resto']>0)]    
    no_both_resto_assign_together=both_resto_assign_together.shape[0]
    TEST_REPORT.append(('Comparative', 'partial_and_New_Resto',
                        'is_any_partial_and_new_resto_schedule', no_both_resto_assign_together!=0))
    TEST_REPORT.append(('Comparative', 'partial_and_New_Resto',
                        '#is_any_partial_and_new_resto_schedule', no_both_resto_assign_together))
    TEST_REPORT.append(('Comparative', 'partial_and_New_Resto',
                        'is_any_partial_and_new_resto_schedule', '<<<new_and_partial_resto_assign_together>>>'))

    both_resto_assign_together.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'new_and_partial_resto_assign_together.csv')),index=False)
    


# In[13]:


def check_new_resto_rule(scheduled_pairs,TEST_REPORT):
    checking_new_restos(scheduled_pairs,TEST_REPORT)
    # NEW Resto C1 -> NO New resto more than 1 for each day
        
    # NEW RESTO C2 -> Partial New Resto < 2 per week for each group
    checking_partial_new_restos(scheduled_pairs,TEST_REPORT)
    # NEW RESTO C3 -> NEW & Partial New Resto are not assigned concurrently
    new_and_partial_resto_assign_together(scheduled_pairs,TEST_REPORT)


# In[14]:


check_new_resto_rule(merlin4_schedules_2,TEST_REPORT)


# In[15]:


TEST_REPORT


# ### COSINE RULE 

# In[16]:


cusine_table = [
    (25, 'Warm Bowls', None, None, None),
    (26, 'Cold Salads', None, None, None),
    (1, 'Arabic & Grills', None, None, None),
    (21, 'Mediterranean Sandwiches & Wraps', 3, 3, 3),
    (27, 'Pasta', None, None, None),
    (22, 'Pizza', 1, 1, 1),
    (17, 'International', 1, 1, 1),
    (24, 'Thai', 3, 3, 3),
    (15, 'Chinese', 3, 3, 3),
    (19, 'Japanese', 2, 2, 2),
    (20, 'Korean', None, None, None),
    (8, 'Indian', 3, 3, 3),
    (9, 'Mexiacan', 3, 3, 3),
    (11, 'Sushi', 2, 2, 2),
    (30, 'Mixed-Bag', 1, 1, 1),
    (28, 'Sandwiches', None, None, None),
    (29, 'Burgers', 2, 2, 2),
]
cusine_table = pd.DataFrame(cusine_table, columns=[
                                    'id', 'name', '3', '4', '5']).set_index('id')
cusine_table.drop(columns=['name'], inplace=True)
cusine_table


# In[17]:


def validate_cusine_rules(scheduled_df, cusine_table, TEST_REPORT):
    
    cusine_table = cusine_table.reset_index()
    cusine_table = pd.melt(cusine_table, id_vars='id', var_name="perday_resto",
                           value_name="max_scadule").rename(columns={'id': 'category'})
    cusine_table.fillna(100, inplace=True)
    cusine_table = cusine_table.astype(float)

    
    cusin_rule_check = scheduled_df.query('schedule_status == 0').groupby(['group_id', 'perday_resto', 'category'])[
        'category'].count().reset_index(name="time_scheduled")
    
    cusin_rule_check = cusin_rule_check.merge(
        cusine_table, on=['perday_resto', 'category'], how='left')

    cusin_rule_check['is_rule_break'] = cusin_rule_check.eval(
        'max_scadule < time_scheduled')

    temp = cusin_rule_check[cusin_rule_check['is_rule_break']]
    temp.to_csv(gapp(os.path.join(processed_base_data_path,
                                  'output', 'validation', 'cusin_level_rule_voilate.csv')))

    is_any_cusin_level_rule_voilate = temp.shape[0]

    TEST_REPORT.append(('Comparative', 'Cusine', 'is_any_cusin_level_rule_voilate',
                        is_any_cusin_level_rule_voilate != 0))
    TEST_REPORT.append(('Comparative', 'Cusine',
                        '#of_cusin_level_rule_voilate', is_any_cusin_level_rule_voilate))
    TEST_REPORT.append(('Comparative', 'Cusine',
                        'cusin_level_rule_voilate', '<<< cusin_level_rule_voilate >>>'))
#%%
validate_cusine_rules(merlin4_schedules_2, cusine_table, TEST_REPORT)


# ### Blackout Test

# In[18]:


def blackout_constraint_voilate(scheduled_df,TEST_REPORT):
    
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


# In[19]:


blackout_constraint_voilate(merlin4_schedules_2,TEST_REPORT)


# ### branches_capacity_violation

# In[20]:


def branch_delivery_capacity(merlin4_schedules_2,TEST_REPORT):
    branch_scheduled = merlin4_schedules_2.query('schedule_status == 0').groupby(['p_date','branch_id'])[
        'branch_id'].count().reset_index(name="time_scheduled")
    
    delivery_cap = merlin4_schedules_2[['p_date','branch_id','delivery_capacity']]
    branch_scheduled=branch_scheduled.merge(delivery_cap,how='inner',on=['p_date','branch_id'])
    
    branch_scheduled['is_rule_break'] = branch_scheduled.eval('delivery_capacity < time_scheduled')
    temp = branch_scheduled[branch_scheduled['is_rule_break']]
    temp.to_csv(gapp(os.path.join(processed_base_data_path,
                                  'output', 'validation', 'branch_capacity_violate.csv')))

    branch_capacity_violate = temp.shape[0]

    TEST_REPORT.append(('Comparative', 'branch_Capacity', 'branch_capacity_violate',
                        branch_capacity_violate != 0))
    TEST_REPORT.append(('Comparative', 'branch_Capacity',
                        '#branch_capacity_violate', branch_capacity_violate))
    TEST_REPORT.append(('Comparative', 'branch_Capacity',
                        'branch_capacity_violate', '<<< branch_capacity_violate >>>'))


# In[21]:


branch_delivery_capacity(merlin4_schedules_2,TEST_REPORT)


# ### Premimum Resto Rule

# In[22]:


def premium_resto_rule(scheduling_file,TEST_REPORT):
#     scheduling_file=scheduling_file[scheduling_file['schedule_status']==0]
#     scheduling_file.reset_index(drop=True,inplace=True)
    
    column_name=['p_date', 'branch_id', 'restaurant_id', 'group_id',
           'is_premium']

    column_name_2=[ 'group_id','perday_resto']

    scheduling_file2=scheduling_file[column_name_2]

    scheduling_file1=scheduling_file[column_name]

    scheduling_file1=scheduling_file1[scheduling_file1['is_premium'].notnull()]
    ssd=scheduling_file1.groupby(['group_id','p_date'])["is_premium"].value_counts().unstack(fill_value=0).reset_index()
    final_df=ssd.merge(scheduling_file2, on=['group_id'], how='left')
    final_df['premimu_resto_flag_rule_break']=np.nan
    final_df.rename(columns={0.0: '#non_premium',1.0: '#premium'},inplace = True)
    
    final_df['premimu_resto_flag_rule_break']=final_df['#premium']/final_df['perday_resto']
    final_df_rule_break_non=final_df[final_df['premimu_resto_flag_rule_break']<0.4]
    print(final_df_rule_break_non.shape)
    
    final_df_rule_break=final_df[final_df['premimu_resto_flag_rule_break']>0.4]
    final_df_rule_break.drop_duplicates(inplace=True)
    final_df_rule_break.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'premium_resto_rule_voilations.csv')), index=False)

    is_any_top_resto_voilate = final_df_rule_break.empty
    no_of_top_resto_voilate = final_df_rule_break.shape[0]
    
    TEST_REPORT.append(('Comparative', 'premium resto rule Voilate',
                        'is any premium resto rule', not is_any_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'premium resto Voilate',
                        '#no_of_any_premium_resto_voilate', no_of_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'Top premium resto Voilate',
                        'premium resto Voilate', '<<< premium resto Voilate >>>'))


# In[23]:


premium_resto_rule(merlin4_schedules_2,TEST_REPORT)


# ### Category Violation

# In[24]:


import collections
def checkIfDuplicates_greater_than_3(listOfElems_resto_id,listOfElems_resto_catgory):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems_resto_id) == len(set(listOfElems_resto_id)):
        if 10 in listOfElems_resto_id and 11 in listOfElems_resto_id:
            return False
        else:
            return True
    else:
        if 10 in listOfElems_resto_id and 11 in listOfElems_resto_id:
            if ([item for item, count in collections.Counter(listOfElems_resto_id).items() if count > 1]==[10] or [item for item, count in collections.Counter(listOfElems_resto_id).items() if count > 1]==[9] or  [item for item, count in collections.Counter(listOfElems_resto_id).items() if count > 1]==[10,9] or [item for item, count in collections.Counter(listOfElems_resto_id).items() if count > 1]==[9,10]):
                    if len(listOfElems_resto_catgory) == len(set(listOfElems_resto_catgory)):
                        return False
                    else:
                        return True
            else:
                return True
        
        else:
            return True
    
def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        if 10 in listOfElems:
            return False
        else:
            return True
    else:
        return True


def rule_category_violation(df_scheduled_with_per_day_resto,TEST_REPORT):
    df_scheduled_with_per_day_resto=df_scheduled_with_per_day_resto[['p_date', 'branch_id', 'restaurant_id', 'group_id',
       'perday_resto', 'type', 'category']]
    
    df_scheduled_with_per_day_resto=df_scheduled_with_per_day_resto.rename(columns={'type': 'restaurant_type','category':'restaurant_category'})
                                                                           
    df_scheduled_with_per_day_resto_less_than_3=df_scheduled_with_per_day_resto[df_scheduled_with_per_day_resto['perday_resto']<=3]

    df_scheduled_with_per_day_resto_greter_than_3=df_scheduled_with_per_day_resto[df_scheduled_with_per_day_resto['perday_resto']>3]

    # 3 or less than 3 per day resto
    df_less_than_3=df_scheduled_with_per_day_resto_less_than_3.groupby(['group_id','p_date','perday_resto'])["restaurant_type"].apply(list).reset_index(name="resto_type_id")

    df_more_than_3_resto_type=df_scheduled_with_per_day_resto_greter_than_3.groupby(['group_id','p_date','perday_resto'])["restaurant_type"].apply(list).reset_index(name="resto_type_id")
    df_more_than_3_resto_category=df_scheduled_with_per_day_resto_greter_than_3.groupby(['group_id','p_date','perday_resto'])["restaurant_category"].apply(list).reset_index(name="restaurant_category")

    final_df_more_than_3_resto_per_day=df_more_than_3_resto_type.merge(df_more_than_3_resto_category, on=['group_id','p_date','perday_resto'], how='left')

    df_less_than_3['category_rule_violate']=np.nan
    for index in range(len(df_less_than_3)):
        listOfElems=df_less_than_3.iloc[index]['resto_type_id']
        df_less_than_3['category_rule_violate'][index]=checkIfDuplicates_1(listOfElems)


    final_df_more_than_3_resto_per_day['category_rule_violate']=np.nan
    for index in range(len(final_df_more_than_3_resto_per_day)):
        listOfElems_resto_id=final_df_more_than_3_resto_per_day.iloc[index]['resto_type_id']
        listOfElems_resto_catgory=final_df_more_than_3_resto_per_day.iloc[index]['restaurant_category']
        final_df_more_than_3_resto_per_day['category_rule_violate'][index]=checkIfDuplicates_greater_than_3(listOfElems_resto_id,listOfElems_resto_catgory)  


    df_final_catgory=pd.concat([df_less_than_3,final_df_more_than_3_resto_per_day],axis=0)
    return_df=df_final_catgory[df_final_catgory['category_rule_violate']==True]
    
    
    return_df.to_csv(gapp(os.path.join(
        processed_base_data_path, 'output', 'validation', 'rule_category_violation.csv')), index=False)

    is_any_top_resto_voilate = return_df.empty
    no_of_top_resto_voilate = return_df.shape[0]
    
    TEST_REPORT.append(('Comparative', 'category rule Voilate',
                        'is any rule category', not is_any_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'premium resto Voilate',
                        '#no of rule category voilate', no_of_top_resto_voilate))
    TEST_REPORT.append(('Comparative', 'rule category Voilate',
                        'rule category Voilate', '<<< rule category Voilate >>>'))


# In[25]:


rule_category_violation(merlin4_schedules_2,TEST_REPORT)


# #### VALIDATION RESULT CSV

# In[26]:


result = pd.DataFrame(TEST_REPORT, columns=[
                      'Scope', 'Domain', 'Test Name', 'Value']).set_index(['Scope', 'Domain'])
result.to_csv(gapp(os.path.join(processed_base_data_path,
                                'output', 'validation', 'validation_result.csv')))


# In[27]:


TEST_REPORT

