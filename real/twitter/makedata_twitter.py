import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import graph_tool.all as gt

from datetime import timedelta

import pickle

import os
import tqdm

import sys
sys.path.append('../../util')

from bsc import calc_paracomp_bern_with_prior_beta
from cython_normterm_discrete import create_fun_with_mem


outdir = './output'

if not os.path.exists(outdir):
    os.mkdir(outdir)

#ground_truths = pd.read_excel('./data/Soccer World Cup 2014 - Ground truth_SpreadSheet.xlsx')
#ground_truths = ground_truths.loc[ground_truths['Event : {Goal, Yellow card, Red card, Penalty shootout}'] != 'Injured', :]
#ground_truths = ground_truths.dropna(how='all').reset_index()
#
#ground_truths_imp = ground_truths.loc[
#               ground_truths['Types of importance'].isin(
#                   ['High importance events', 'HIgh importance events']), :]
#ground_truths_imp['Game Nos.'] = ground_truths_imp['Game Nos.'].ffill()
#ground_truths_imp['Game Nos.'] = ground_truths_imp['Game Nos.'].astype(int)
#datetime_start_end = pd.DataFrame({
#              'start': ground_truths_imp.groupby('Game Nos.')['Date:Time in UTC Format'].first(),
#              'end': ground_truths_imp.groupby('Game Nos.')['Date:Time in UTC Format'].last()})

df = pd.read_csv('./data/Twitter_WorldCup_2014_resolved.txt', names=['date', 'entity1', 'entity2'], sep=' ')
df['date'] = pd.to_datetime(df['date'], format='%m:%d:%Y:%H:%M:%S')

count_od = df.groupby([pd.Grouper(key='date', freq='1h'), 'entity1', 'entity2']).size().reset_index().rename(columns={0: 'count'})

count_od = count_od.loc[(count_od['date'] >= pd.to_datetime('2014-06-01 00:00:00')) & (count_od['date'] <= pd.to_datetime('2014-07-15 23:00:00')), :]

terms_array = sorted(count_od['date'].unique())
entities_array = pd.concat([df['entity1'], df['entity2']]).unique()

infile_entities_dict_name2id = './data/entities_dict_id2name.pkl'
infile_entities_dict_id2name = './data/entities_dict_name2id.pkl'

if os.path.exists(infile_entities_dict_name2id) & os.path.exists(infile_entities_dict_id2name):
    with open(infile_entities_dict_name2id, 'rb') as f:
        entities_dict_name2id = pickle.load(f)
    with open(infile_entities_dict_id2name, 'rb') as f:
        entities_dict_id2name = pickle.load(f)
else:
    entities_dict_name2id = {v: i for i, v in enumerate(entities_array)}
    entities_dict_id2name = {i: v for i, v in enumerate(entities_array)}
    
    with open(infile_entities_dict_name2id, 'wb') as f:
        pickle.dump(entities_dict_name2id, f)
    with open(infile_entities_dict_id2name, 'wb') as f:
        pickle.dump(entities_dict_id2name, f)

count_od['entity1'] = [entities_dict_name2id[o] for o in count_od['entity1']]
count_od['entity2'] = [entities_dict_name2id[o] for o in count_od['entity2']]

with open('./data/count_od_by_week.pkl', 'wb') as f:
    pickle.dump(count_od, f)

with open('./data/terms_array.pkl', 'wb') as f:
    pickle.dump(terms_array, f)
