import re
import pandas as pd
import numpy as np
import zipfile
from keras.models import load_model
import pickle
import subprocess
from functools import partial
import tempfile
import joblib

def __make_temp_dir():
    try:
        temp_dir = tempfile.mkdtemp()     
        print("Temporary directory created:", temp_dir)
        return temp_dir
    
    except Exception as e:
        print("An error has occurred.:", e)
        return None

def __calculate_average_atomic_mass(molecular_formula):
    # Define mass
    average_atomic_masses = {'H': 1.007825, 'C': 12.000000, 'O':15.994915}
    # Calculate atomic mass from molecular formula
    elements_with_counts = re.findall(r'([A-Z][a-z]*)(\d*)', molecular_formula)
    element_counts = {element[0]: int(element[1]) if element[1] else 1 for element in elements_with_counts}
    average_atomic_mass = sum(element_counts[element] * average_atomic_masses[element] for element in element_counts)
    return average_atomic_mass

def __cal_mod(averagemz):
    num = ((averagemz % __calculate_average_atomic_mass('CH2')) % __calculate_average_atomic_mass('H2')) % (__calculate_average_atomic_mass('H14') % __calculate_average_atomic_mass('CH2')) 
    return num

def __wide_spectrum(df):
    df_wide_spct = pd.DataFrame(columns=range(1, 1251), index=None)

    for i, row in df.iterrows():
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
        df_wide_spct.loc[i , xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())
    df_wide_spct = df_wide_spct.fillna(0).infer_objects()

    return df_wide_spct

def __wide_neulloss(df):
    df_wide_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)

    for i, row in df.iterrows():
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
        df_wide_neuloss.loc[i , xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values()) 
    df_wide_neuloss = df_wide_neuloss.fillna(0).infer_objects()

    return df_wide_neuloss

def __spectrum_neutralloss_table(df): 

    df_wide_spct = __wide_spectrum(df)
    df_wide_neuloss = __wide_neulloss(df)

    df_data = pd.concat([df_wide_spct, df_wide_neuloss, df[['Average Mz', 'MCHvalue']]], axis=1).drop(columns=['Average Mz'])
    
    return df_data  

def __msp2df(file_path):

    msp_data = pd.read_csv(file_path, header=None).rename(columns={0:'x'})
    msp_data['name'] = msp_data['x'].apply(lambda x: x.split(':')[0] if ': ' in x else 'MSMSspectrum')
    msp_data['value'] = msp_data['x'].apply(lambda x: x.split(':', 1)[1] if ':' in x else x).replace('\t', ':', regex=True).\
        apply(lambda x: x.split('|')[0] if '|' in x else x)
    msp_data['id'] = (msp_data['name'] == 'NAME').cumsum() - 1

    msp_data_ = msp_data.pivot_table(index='id', columns='name', values='value', aggfunc=' '.join).reset_index()
    msp_data_.columns = msp_data_.columns.str.lower()
    makedf = msp_data_[['id','name','ontology','precursormz','msmsspectrum']].\
        rename(columns={'id':'Alignment ID', 'name':'Metabolitename','ontology':'Ontology','precursormz':'Average Mz', 'msmsspectrum':'MS/MS spectrum'})
    
    return makedf

def __import_data(path, format = None, ms2spc_name = 'MS/MS spectrum', prec_name = 'Average Mz', ID = 'Alignment ID'):

    data_type = path.split('.')[-1]

    if data_type == 'csv':
        df = pd.read_csv(path)
        df = df[[ms2spc_name,prec_name,ID]].rename(columns={ms2spc_name:'MS/MS spectrum',prec_name:'Average Mz', ID:'Alignment ID'})

    elif data_type == 'msp':
        df = __msp2df(path)
        df = df[[ms2spc_name,prec_name,ID]].rename(columns={ms2spc_name:'MS/MS spectrum',prec_name:'Average Mz', ID:'Alignment ID'})
        df['Average Mz'] = df['Average Mz'].astype(float)
        
    elif data_type == 'txt' and format == None:
        df = pd.read_csv(path, sep='\t', delimiter=None)
        df = df[[ms2spc_name,prec_name,ID]].rename(columns={ms2spc_name:'MS/MS spectrum',prec_name:'Average Mz', ID:'Alignment ID'})

    elif data_type == 'txt' and format =='MSDIAL':
        df = pd.read_csv(path, sep='\t', header=4, delimiter=None)
        df = df[[ms2spc_name,prec_name,ID]].rename(columns={ms2spc_name:'MS/MS spectrum',prec_name:'Average Mz', ID:'Alignment ID'})

    else:
        print('Data type not supported')

    return df

def __make_table(path, format = None, ms2spc_name = 'MS/MS spectrum', prec_name = 'Average Mz', ID = 'Alignment ID'):
    df = __import_data(path, format = format, ms2spc_name = ms2spc_name, prec_name = prec_name, ID=ID)
    df.loc[:, 'MCHvalue'] = __cal_mod(df['Average Mz'])
    df_data = __spectrum_neutralloss_table(df)  

    return df, df_data

def __import_zenodo_data():
    filename = "10.5281/zenodo.13893001"
    subprocess.run(["python", "-m", "zenodo_get", filename])

def __unzip_file(temp_dir):
    zip_file = 'model_data.zip' 

    extract_to = temp_dir ###
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def __import_and_unzip_zenodo_data(temp_dir):
    __import_zenodo_data()
    __unzip_file(temp_dir)

def __load_models(temp_dir, ionmode = 'negative'): ###
    #select ion mode: negative or positive
    if ionmode == 'negative':
        model = joblib.load(f'{temp_dir}/model_data/negative/model.joblib')
        model_column = pickle.load(open(f'{temp_dir}/model_data/negative/column.pkl', 'rb'))
        label_encoder_neg = pickle.load(open(f'{temp_dir}/model_data/negative/dict.pkl', 'rb'))
        modelclass_replacement = {index: label for index, label in enumerate(label_encoder_neg)}
        print('ion mode: negative')

    elif ionmode == 'positive':   
        model = joblib.load(f'{temp_dir}/model_data/positive/model.joblib')
        model_column = pickle.load(open(f'{temp_dir}/model_data/positive/column.pkl', 'rb'))
        label_encoder_pos = pickle.load(open(f'{temp_dir}/model_data/positive/dict.pkl', 'rb'))
        modelclass_replacement = {index: label for index, label in enumerate(label_encoder_pos)}
        print('ion mode: positive')

    else:
        print('Please select ion mode: negative or positive')

    return model, model_column, modelclass_replacement

def __apply_replacement(x,y):
    return y.get(x, x)

def ___create_pred_class(row, threshold=1):
    classes = [f"{col}:{round(value * 100, 1)}%" for col, value in sorted(row.items(), key=lambda x: x[1], reverse=True) if value >= threshold/100]
    return ','.join(classes)

def __pred_class(df, df_data, modelclass_replacement, model_column, model, threshold=1):
    
    df_data.columns = df_data.columns.astype(str)
    X1_test = df_data[model_column[0]].values
    y_pred_test = model.predict_proba(X1_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)

    y_pred_test_max_ = np.vectorize(__apply_replacement)(y_pred_test_max, modelclass_replacement)

    df_class_num = pd.DataFrame({'class': list(modelclass_replacement.values()), 'num': list(modelclass_replacement.keys())})
    df_test_predclass = pd.DataFrame(y_pred_test).rename(columns=df_class_num['class'])

    create_pred_class_partial = partial(___create_pred_class, threshold=threshold)

    df_test_predclass['predict_1class'] = y_pred_test_max_
    df_test_predclass['predict_candidateclass'] = df_test_predclass.drop('predict_1class', axis=1).apply(create_pred_class_partial, axis=1)

    df_pred_result = df[['MS/MS spectrum', 'Average Mz']].reset_index(drop=True).\
        merge(df_test_predclass[['predict_1class', 'predict_candidateclass']], left_index=True, right_index=True)
    df_pred_result.insert(0, 'ID', df['Alignment ID'])

    return df_pred_result

def __save_pred_result(df_test_predclass, exppath = 'pred_result.csv'):
    df_test_predclass.to_csv(exppath, index=False)

def predclass(
    path,
    format = None, 
    ms2spc_name = 'MS/MS spectrum', 
    prec_name = 'Average Mz',
    ID = 'Alignment ID',
    ionmode = 'negative', 
    threshold=1, 
    exppath = None,
):
    
    """\
    This function is to predict lipid class from LC-MS MS/MS spectrum and precursor ion value.
    This version is supporting total 97 lipid sub classes.
    
    Parameters
    ----------
    path
        A path which contains MS/MS spectrum and precursor ion data.
    format
        if you use 'MSDIAL' export format, set 'MSDIAL'. Default is None.
    ms2spc_name
        column name of MS/MS spectrum value.
    prec_name
        column name of precursor ion value.
    ID
        column name of sample ID.
    ionmode
        Ion mode, either 'negative' / 'positive'. Default is 'negative'.
    threshold
        A value of Minimum probability to be considered as a prediction class. Default is 1%.
    exppath
        Path to save the predicted result. Default is None. 
    """

    if 'temp_dir' in locals():
        None
    else: 
        temp_dir = __make_temp_dir()
        __import_and_unzip_zenodo_data(temp_dir)

    model, model_column, modelclass_replacement = __load_models(temp_dir, ionmode=ionmode) 
    df, df_data = __make_table(path, format = format, ms2spc_name = ms2spc_name, prec_name = prec_name, ID=ID)
    df_test_predclass = __pred_class(df, df_data, modelclass_replacement, model_column, model, threshold=threshold)

    if exppath == None:
        return df_test_predclass
    else:
        __save_pred_result(df_test_predclass, exppath = exppath)


def __correctclass_data(path, format = None, class_name = 'ontology'):

    data_type = path.split('.')[-1]
    
    if data_type == 'csv':
        cdf = pd.read_csv(path)
        cdf = cdf[[class_name]].rename(columns={class_name:'ontology'})

    elif data_type == 'msp':
        cdf = __msp2df(path)
        cdf = cdf[[class_name]].rename(columns={class_name:'ontology'})
        
    elif data_type == 'txt' and format == None:
        cdf = pd.read_csv(path, sep='\t', delimiter=None)
        cdf = cdf[[class_name]].rename(columns={class_name:'ontology'})

    elif data_type == 'txt' and format =='MSDIAL':
        cdf = pd.read_csv(path, sep='\t', header=4, delimiter=None)
        cdf = cdf[[class_name]].rename(columns={class_name:'ontology'})

    else:
        print('Data type not supported')

    return cdf


def __candidate_search(df_test_predclass, path, format = None, class_name = 'ontology'):
    df_test_predclass[['correct_class']] = __correctclass_data(path, format = format, class_name = class_name)[['ontology']]
    df_test_predclass['correct_class'] = df_test_predclass['correct_class'].str.replace(' ', '')
    
    #make candidate class list
    all_candidate_list = []
    for all_canf in df_test_predclass.predict_candidateclass:
        pred_class_list = all_canf
        candidate_list = []
        for candidate in pred_class_list.split(','):
            parts = candidate.split(':')[0]
            candidate_list.append(parts)
        all_candidate_list.append(candidate_list)

    #eval correct class in candidate list
    list_candidate = []
    for i in range(len(df_test_predclass)):
        if df_test_predclass['correct_class'][i] in all_candidate_list[i]:
            answer = 'o'
        else:
            answer = 'x'
        list_candidate.append(answer)

    df_test_predclass[['predict_candidateclass']] = pd.DataFrame(list_candidate)

    return list_candidate,df_test_predclass


def prediction_summary(
    df_test_predclass :pd.DataFrame, 
    path, format = None, class_name = 'ontology',
    showdf = False,
    exppath = None,
):
    
    """\
    By default, the summary result is printed. If you want to return the result as a dataframe, 
    you can set the df parameter to True.

    Parameters
    ----------
    df_test_predclass
        Predicted result represented as a Pandas DataFrame.
        This is the output of the predclass function.
    path
        A path which contains the correct class data.
    format
        if you use 'MSDIAL' format, set 'MSDIAL'. Default is None.
    class_name
        Correct class column name. Default is 'ontology'.
    showdf
        Show a dataframe which predicted result of each sample. Default is False. 
    exppath
        Path to save the dataframe. Default is None.
    """

    list_candidate, df_test_predclass_eval = __candidate_search(df_test_predclass, path, format = format, class_name = class_name)

    pred_1class_eval = len(df_test_predclass_eval[df_test_predclass_eval['predict_1class'] == df_test_predclass_eval['correct_class']])
    pred_cand_eval = list_candidate.count('o')
    mispred = list_candidate.count('x')

    print(f'pred_1class_eval: {pred_1class_eval}')
    print(f'pred_cand_eval: {pred_cand_eval}')
    print(f'mispred: {mispred}')

    print(f'pred 1class correct ratio: {pred_1class_eval/len(df_test_predclass_eval)}')
    print(f'pred candidate correct ratio: {pred_cand_eval/len(df_test_predclass_eval)}')
    print(f'mispred ratio: {mispred/len(df_test_predclass_eval)}')

    if showdf == False: 
        return #show summary
    
    else:
        if exppath == None: #show df 
            return df_test_predclass_eval
        else:  #save df
            df_test_predclass_eval.to_csv(exppath, index=False)
            return df_test_predclass_eval
        
__all__ = ['predclass', 'prediction_summary']