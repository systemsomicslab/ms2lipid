import pandas as pd
import tensorflow as tf
import numpy as np
import sqlite3
import random
import torch
import re
import keras
from keras import layers
from keras.utils import to_categorical
import openpyxl

#read dataset
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

def loadDataframe(path):
    try:
        if path.endswith('.txt'):
            _td = pd.read_table(path, header=4, delimiter=None)

    except Exception as e:
        print(f"Error processing file: {path}")
        print(e)

    #remove na
    td = _td.filter(items=['MS/MS spectrum','Average Rt(min)','Annotation tag (VS1.0)', 'Average Mz','Reference RT','Reference m/z','Adduct type','Ontology',\
                           'Metabolite name','Alignment ID','Comment','Formula','INCHIKEY','SMILES']).dropna(how='any')
    #remove unknown
    td = td[~td.isin(['Unknown']).any(axis=1)]
    #annotation tag / MS/MS matched <= 420
    td['Annotation tag (VS1.0)'] = pd.to_numeric(td['Annotation tag (VS1.0)'], errors='coerce',downcast='integer')
    td = td.dropna()
    td['Annotation tag (VS1.0)'] = pd.to_numeric(td['Annotation tag (VS1.0)'], errors='coerce',downcast='integer').astype(float)
    td = td.query("`Annotation tag (VS1.0)` <= 420").reset_index(drop=True)   
    td.insert(0, 'dataset', path) 
    td.insert(1, 'sampleid', range(1, len(td) + 1))
    return td 

def loadDataframe_for_RL(path):
    try:
        if path.endswith('.xlsx'):
            _td = pd.read_excel(path, header=9)

    except Exception as e:
        print(f"Error processing file: {path}")
        print(e)

    #remove na
    td = _td.filter(items=['MS/MS spectrum','Average Rt(min)','Annotation tag (VS1.0)', 'Average Mz','Reference RT','Reference m/z','Adduct type','Ontology',\
                           'Metabolite name','Alignment ID','Comment','Formula','INCHIKEY','SMILES'])#.dropna(how='any')
    #remove unknown
    td = td[~td.isin(['Unknown']).any(axis=1)]
    #annotation tag / MS/MS matched <= 420
    td.insert(0, 'dataset', path) 
    td.insert(1, 'sampleid', range(1, len(td) + 1))
    return td 


def insertdata(td, path = '../../data/basedata/ms2_lipid_data.db'):  
    td.columns = td.columns.str.replace('/', '')
    td.columns = td.columns.str.replace(' ', '')
    td['sampleid'] = td['sampleid'].astype(str)
    td['AlignmentID'] = td['AlignmentID'].astype(float)
    # connect SQLite
    with sqlite3.connect(path) as conn:      
        c = conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')
        #if the count is 1, then table exists
        if c.fetchone()[0]!=1 : {           
            # make table
            conn.execute('''
                 CREATE TABLE data (
                    dataset TEXT,
                    sampleid TEXT,
                    MSMSspectrum TEXT,
                    "Annotationtag(VS1.0)" INTEGER,
                    "AverageRt(min)" REAL,
                    AverageMz REAL,
                    ReferenceRT REAL,
                    Referencemz REAL,
                    Adducttype TEXT,
                    Ontology TEXT,
                    Metabolitename TEXT,
                    AlignmentID INTEGER,
                    Comment TEXT,
                    Formula TEXT,
                    INCHIKEY TEXT,
                    SMILES TEXT
                );
            ''')
                   	
        }
        # insert data
        col_names = ', '.join(['`{}`'.format(c) for c in td.columns])
        placeholders = ', '.join(['?' for _ in td.columns])
        insert_query = 'INSERT INTO data ({}) VALUES ({})'.format(col_names, placeholders)
        conn.executemany(insert_query, td.to_records(index=False))

#wide data 
def import_sqlite3_data(i):
    conn = sqlite3.connect(i)
    df = pd.read_sql_query("SELECT * FROM data", conn)
    conn.close()  
    return df 



def spectrum_tidy(df_pos_s):#MS/MS column to wide
    df_pos_exp = pd.DataFrame(columns=range(1, 1251))
    df_pos_exp[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']] = df_pos_s[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']]
    df_pos_exp = df_pos_exp.set_index(['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']).sort_index()
    for i, row in df_pos_s.iterrows():
        data = row['MSMSspectrum']
        split_data = data.split(" ")
        data_list = [i.split(":") for i in split_data]  
        xy_data = {}
        for item in data_list:
            if len(item) != 2:
                continue               
            mz, intensity = item
            if round(float(mz)) > 1250:
                continue               
            if int(round(float(mz))) in xy_data:
                xy_data[int(round(float(mz)))] += float(intensity)
            else:
                xy_data[int(round(float(mz)))] = float(intensity)   
        df_pos_exp.loc[tuple(row[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())    
    df_pos_exp = df_pos_exp.fillna(0)
    df_pos_exp_ = df_pos_exp.loc[:, (df_pos_exp != 0).any(axis=0)]   
    return df_pos_exp_

def neutralloss_tidy(df_pos_s):#neutralloss from precursor
    df_pos_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)
    df_pos_neuloss[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']] = df_pos_s[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']]
    df_pos_neuloss = df_pos_neuloss.set_index(['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']).sort_index()    
    for i, row in df_pos_s.iterrows():
        data = row['MSMSspectrum']
        split_data = data.split(" ")
        data_list = [i.split(":") for i in split_data]
        precursor = row['AverageMz']      
        xy_data = {}
        for item in data_list:           
            if len(item) != 2:
                continue               
            mz, intensity = item  
            neutralloss =  - float(precursor) + float(mz)         
            if round(float(neutralloss)) < -1250:
                continue             
            if int(round(float(neutralloss))) in xy_data:
                xy_data[int(round(float(neutralloss)))] += float(intensity)
            else:
                xy_data[int(round(float(neutralloss)))] = float(intensity)   
        df_pos_neuloss.loc[tuple(row[['Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values()) 
    df_pos_neuloss = df_pos_neuloss.fillna(0)
    df_pos_neuloss_ = df_pos_neuloss.loc[:, (df_pos_neuloss != 0).any(axis=0)]  
    return df_pos_neuloss_

def spectrum_neutralloss_table(df_neg_exp_, df_neg_neuloss_): #bond spectrum and neuloss data
    #spectrum: >70
    collist = df_neg_exp_.columns.astype(int).tolist()
    selected_collist_ =[num for num in collist if abs(num) >= 70]
    df_neg_exp_2 =df_neg_exp_[selected_collist_]
    #NL: >10
    collist = df_neg_neuloss_.columns.astype(int).tolist()
    selected_collist =[num for num in collist if abs(num) >= 10]
    df_neg_neuloss_2 =df_neg_neuloss_[selected_collist]   
    df_data = pd.concat([df_neg_exp_2, df_neg_neuloss_2], axis=1).reset_index()   
    return df_data
