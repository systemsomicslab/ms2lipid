import pandas as pd
import tensorflow as tf
import numpy as np

import sqlite3
import random
import torch

import re

import keras
from keras import layers



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)


def loadDataframe(path):
    _td = pd.read_table(path, header=4, delimiter=None)   
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


def insertdata(td):  
    td.columns = td.columns.str.replace('/', '')
    td.columns = td.columns.str.replace(' ', '')
    td['sampleid'] = td['sampleid'].astype(str)
    td['AlignmentID'] = td['AlignmentID'].astype(float)
    # connect SQLite
    with sqlite3.connect('Data/01_sqlite_data/ms2_lipid2.db') as conn:      
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


def loadDataframe_rl(path):   
    _td = pd.read_excel(path, index_col=None, header=9) 
    td = _td.filter(items=['MS/MS spectrum','Average Rt(min)','Annotation tag (VS1.0)', 'Average Mz','Reference RT','Reference m/z','Adduct type','Ontology',\
                           'Metabolite name','Alignment ID','Comment','Formula','INCHIKEY','SMILES','MS/MS assigned'])
    td = td[td['MS/MS assigned'] & ~td.isin(['Unknown']).any(axis=1) & ~td['Metabolite name'].str.contains("SPLASH")].drop('MS/MS assigned', axis=1)       
    td.insert(0, 'dataset', path) 
    td.insert(1, 'sampleid', range(1, len(td) + 1))
    return td 


def insertdata_rl(td):
    td.columns = td.columns.str.replace('/', '')
    td.columns = td.columns.str.replace(' ', '')
    td['sampleid'] = td['sampleid'].astype(str)
    td['AlignmentID'] = td['AlignmentID'].astype(float)

    with sqlite3.connect('Data/01_sqlite_data/ms2_rikentest2.db') as conn:
        
        c = conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')

        if c.fetchone()[0]!=1 : {

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

        col_names = ', '.join(['`{}`'.format(c) for c in td.columns])
        placeholders = ', '.join(['?' for _ in td.columns])
        insert_query = 'INSERT INTO data ({}) VALUES ({})'.format(col_names, placeholders)
        conn.executemany(insert_query, td.to_records(index=False))


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


def spectrum_tidy_rl(df_pos_s):#MS/MS column to wide
    df_pos_exp = pd.DataFrame(columns=range(1, 1251))
    df_pos_exp[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']] = df_pos_s[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']]
    df_pos_exp = df_pos_exp.set_index(['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']).sort_index()
    
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
    
        df_pos_exp.loc[tuple(row[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_exp = df_pos_exp.fillna(0)

    return df_pos_exp
    
    
def neutralloss_tidy_rl(df_pos_s):#neutralloss from precursor
    df_pos_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)
    df_pos_neuloss[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']] = df_pos_s[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']]
    df_pos_neuloss = df_pos_neuloss.set_index(['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']).sort_index()
    
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
     
        df_pos_neuloss.loc[tuple(row[['id','Metabolitename', 'Ontology', 'dataset', 'AlignmentID','AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_neuloss = df_pos_neuloss.fillna(0)
 
    return df_pos_neuloss


def spectrum_neutralloss_table_rl(df_neg_exp_, df_neg_neuloss_):
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


def spectrum_tidy_f(df_pos_s):#MS/MS column to wide
    df_pos_exp = pd.DataFrame(columns=range(1, 1251))
    df_pos_exp[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']] = df_pos_s[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']]
    df_pos_exp = df_pos_exp.set_index(['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']).sort_index()
    
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
    
        df_pos_exp.loc[tuple(row[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_exp = df_pos_exp.fillna(0)
   
    return df_pos_exp
    
    
def neutralloss_tidy_f(df_pos_s):#neutralloss from precursor
    df_pos_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)
    df_pos_neuloss[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']] = df_pos_s[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']]
    df_pos_neuloss = df_pos_neuloss.set_index(['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']).sort_index()
    
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
     
        df_pos_neuloss.loc[tuple(row[['ID', 'Ontology', 'Metabolitename', 'MSMSspectrum', 'AverageMz']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_neuloss = df_pos_neuloss.fillna(0)
    
    return df_pos_neuloss


def spectrum_neutralloss_table_f(df_neg_exp_, df_neg_neuloss_):
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


def spectrum_tidy_f2(df_pos_s):#MS/MS column to wide
    df_pos_exp = pd.DataFrame(columns=range(1, 1251))
    df_pos_exp[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']] = df_pos_s[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']]
    df_pos_exp = df_pos_exp.set_index(['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']).sort_index()
    
    for i, row in df_pos_s.iterrows():
        data = row['MS/MS spectrum']
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
    
        df_pos_exp.loc[tuple(row[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_exp = df_pos_exp.fillna(0)
   
    return df_pos_exp
    
    
def neutralloss_tidy_f2(df_pos_s):#neutralloss from precursor
    df_pos_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)
    df_pos_neuloss[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']] = df_pos_s[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']]
    df_pos_neuloss = df_pos_neuloss.set_index(['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']).sort_index()
    
    for i, row in df_pos_s.iterrows():
        data = row['MS/MS spectrum']
        split_data = data.split(" ")
        data_list = [i.split(":") for i in split_data]
        precursor = row['Average Mz']
        
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
     
        df_pos_neuloss.loc[tuple(row[['Alignment ID', 'Average Mz', 'Metabolite name', 'Ontology']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    
    df_pos_neuloss = df_pos_neuloss.fillna(0)
   
    return df_pos_neuloss


def spectrum_neutralloss_table_f2(df_neg_exp_, df_neg_neuloss_):
    #70以上のスペクトルを選択
    collist = df_neg_exp_.columns.astype(int).tolist()
    selected_collist_ =[num for num in collist if abs(num) >= 70]
    df_neg_exp_2 =df_neg_exp_[selected_collist_]
    
    #10以上のニュートラルロスを選択
    collist = df_neg_neuloss_.columns.astype(int).tolist()
    selected_collist =[num for num in collist if abs(num) >= 10]
    df_neg_neuloss_2 =df_neg_neuloss_[selected_collist]
    
    df_data = pd.concat([df_neg_exp_2, df_neg_neuloss_2], axis=1).reset_index()
    
    return df_data


def calculate_average_atomic_mass(molecular_formula):
    # Define mass
    average_atomic_masses = {'H': 1.007825, 'C': 12.000000, 'O':15.994915}
    # Calculate atomic mass from molecular formula
    elements_with_counts = re.findall(r'([A-Z][a-z]*)(\d*)', molecular_formula)
    element_counts = {element[0]: int(element[1]) if element[1] else 1 for element in elements_with_counts}
    average_atomic_mass = sum(element_counts[element] * average_atomic_masses[element] for element in element_counts)
    return average_atomic_mass


def cal_mod(averagemz):
    num = ((averagemz % calculate_average_atomic_mass('CH2')) % calculate_average_atomic_mass('H2')) % (calculate_average_atomic_mass('H14') % calculate_average_atomic_mass('CH2')) 
    return num


def neg_build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=4224,
                                            step=128),
                               activation='relu'))
        if hp.Choice('batchnorm_and_dropout', ['batch', 'dropout', 'both']) == 'batch':
            model.add(layers.BatchNormalization())
        elif hp.Choice('batchnorm_and_dropout', ['batch', 'dropout', 'both']) == 'dropout':
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(layers.Dropout(dropout_rate))
        else:
            model.add(layers.BatchNormalization())
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(69, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
    return model
    

def pos_build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=4224,
                                            step=128),
                               activation='relu'))
        if hp.Choice('batchnorm_and_dropout', ['batch', 'dropout', 'both']) == 'batch':
            model.add(layers.BatchNormalization())
        elif hp.Choice('batchnorm_and_dropout', ['batch', 'dropout', 'both']) == 'dropout':
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(layers.Dropout(dropout_rate))
        else:
            model.add(layers.BatchNormalization())
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(63, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
    return model


def replacement_dict_mapping(class_list):
    replacement_dict = {class_name: i for i, class_name in enumerate(class_list)}
    return replacement_dict


def check_ontology_in_pred_class(row):
    ontology = row['Ontology']
    pred_class = row['pred_class']
    return ontology in pred_class


def create_pred_class(row):
    classes = [f"{col}:{round(value, 3)}" for col, value in sorted(row.items(), key=lambda x: x[1], reverse=True) if value >= 0.01]
    return ','.join(classes)


def apply_replacement(x,y):
    return y.get(x, x)
    

def cal_accscore(df_neg_data_sel1, replacement_dict, loaded_negcolumns, loaded_negmodel):

    X1_test = df_neg_data_sel1[loaded_negcolumns.tolist()].values
    y1_test = df_neg_data_sel1['Ontology']
    from keras.utils import to_categorical
    y1_test_replaced = y1_test.replace(replacement_dict)

    replacement_dict_num = replacement_dict_mapping(y1_test_replaced.drop_duplicates().tolist())
    y1_test_replaced_num = y1_test_replaced.replace(replacement_dict_num)
    y1_test_onehot = to_categorical(y1_test_replaced_num, len(y1_test_replaced_num.drop_duplicates()))

    y_pred_test = loaded_negmodel.predict(X1_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)
    _, y_test_acc = np.where(y1_test_onehot > 0)
        
    replacement_dict_reversed_num = {value: key for key, value in replacement_dict_num.items()}
    replacement_dict_reversed = {value: key for key, value in replacement_dict.items()}

    y_pred_test_max_ = np.vectorize(apply_replacement)(y_pred_test_max, replacement_dict_reversed)
    y_test_acc_ = np.vectorize(apply_replacement)(np.vectorize(apply_replacement)(y_test_acc, replacement_dict_reversed_num), replacement_dict_reversed)

    replacement_dict_reversed_num_df = pd.DataFrame(columns=['re_replaced_name', 'replaced_name'])
    for replacement, original in replacement_dict_reversed_num.items():
        replacement_dict_reversed_num_df = pd.concat([replacement_dict_reversed_num_df, pd.DataFrame({'re_replaced_name': [replacement], 'replaced_name': [original]})], ignore_index=True)

    replacement_dict_reversed_df = pd.DataFrame(columns=['replaced_name', 'name'])
    for replacement, original in replacement_dict_reversed.items():
        replacement_dict_reversed_df = pd.concat([replacement_dict_reversed_df, pd.DataFrame({'replaced_name': [replacement], 'name': [original]})], ignore_index=True)

    df_ontname = pd.merge(replacement_dict_reversed_num_df, replacement_dict_reversed_df, on='replaced_name', how='right')   
    df_test_predclass = pd.DataFrame(y_pred_test).rename(columns=df_ontname['name'])

    df_test_predclass['pred_1class'] = y_pred_test_max_
    df_test_predclass['pred_class'] = df_test_predclass.drop('pred_1class', axis=1).apply(create_pred_class, axis=1)
    df_test_predclass['pred_number'] = df_test_predclass['pred_class'].apply(lambda x: len(x.split(',')))
        
    df_test_predclass_v = df_test_predclass[['pred_1class','pred_class','pred_number']]
    df_copy = df_neg_data_sel1.reset_index().rename(columns={'index':'idx'})['idx'].copy(deep=True)
    df_test_predclass_v = df_test_predclass_v.assign(idx=df_copy)
        
    df_mispred = pd.concat([df_test_predclass_v.set_index('idx')[y_test_acc_ != y_pred_test_max_].reset_index(drop=True),\
                pd.DataFrame(df_neg_data_sel1[y_test_acc_ != y_pred_test_max_]).reset_index().rename(columns={'index':'id'})],axis=1)
        
    df_mispred['TF'] = df_mispred.apply(check_ontology_in_pred_class, axis=1)
        
    pred1st =  len(df_neg_data_sel1[y_test_acc_ == y_pred_test_max_]) 
    predcandidate = (len(df_neg_data_sel1[y_test_acc_ == y_pred_test_max_]) + len(df_mispred.query('TF == True')))
    num = len(df_test_predclass)

    return pred1st, predcandidate, num


def get_fileid(path):
    path_elements = path.split("/")
    file_name = path_elements[-1]
    file_name_elements = file_name.split("_")
    return file_name_elements[0]


def read_MSP(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_name = []
    data_mz = []
    data_ad = []
    data_f = []
    
    for line in lines:
        if line.startswith('NAME'):
            name_data = line.split(': ')
            data_name.append({'NAME': name_data[1]}) 
    
        elif line.startswith('PRECURSORMZ'):
            mz_data = line.split() 
            data_mz.append({'MZ': mz_data[1]})
    
        elif line.startswith('PRECURSORTYPE'):
            ad_data = line.split() 
            data_ad.append({'Adduct': ad_data[1]})
    
        elif line.startswith('FORMULA'):
            f_data = line.split() 
            data_f.append({'Formula': f_data[1]})
            
    df = pd.concat([pd.DataFrame(data_name), pd.DataFrame(data_mz), pd.DataFrame(data_ad), pd.DataFrame(data_f)], axis=1)

    return df