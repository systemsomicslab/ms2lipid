import re
import pandas as pd
import numpy as np
import zipfile
from keras.models import load_model
import pickle
import subprocess
from functools import partial
import os
import tempfile
from pandas import Series
import os

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
        data = row['ms2spectrum']     
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
    df_wide_spct = df_wide_spct.fillna(0) 

    return df_wide_spct

def __wide_neulloss(df):
    df_wide_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)

    for i, row in df.iterrows():
        data = row['ms2spectrum']
        split_data = data.split(" ")
        data_list = [i.split(":") for i in split_data]
        precursor = row['precursorion']      
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
    df_wide_neuloss = df_wide_neuloss.fillna(0)

    return df_wide_neuloss

def __spectrum_neutralloss_table(df): 

    df_wide_spct = __wide_spectrum(df)
    df_wide_neuloss = __wide_neulloss(df)

    df_data = pd.concat([df_wide_spct, df_wide_neuloss, df[['precursorion', 'mod2_value', 'mch_value']]], axis=1).drop(columns=['precursorion'])
    
    return df_data  

def __make_table(ms2spectrum, precursorion):
    df = pd.DataFrame({'ms2spectrum': ms2spectrum, 'precursorion': precursorion})
    df['mod2_value'] = df['precursorion'].round().astype(int) % 2
    df['mch_value'] = __cal_mod(df['precursorion'])
    df_data = __spectrum_neutralloss_table(df)  

    return df, df_data

def __import_zenodo_data(mode = 'x'):
    if mode == 'test':
        filename = "10.5281/zenodo.10674847"
        subprocess.run(["python", "-m", "zenodo_get", filename])

    else:
        filename = "10.5281/zenodo.10674847" ###
        subprocess.run(["python", "-m", "zenodo_get", filename]) ###

def __unzip_file(temp_dir, mode = 'x'):
    if mode == 'test':
        zip_file = 'testdata.zip' 
    else:
        zip_file = 'testdata.zip' ###

    extract_to = os.path.join(temp_dir) ###
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def __import_and_unzip_zenodo_data(temp_dir, mode='x'):
    __import_zenodo_data(mode = mode)
    __unzip_file(temp_dir, mode = mode)

def __load_models(temp_dir, ionmode = 'negative'): ###
    #select ion mode: negative or positive
    if ionmode == 'negative':
        model = load_model(f'{temp_dir}/testdata/model')
        model_column = pickle.load(open(f'{temp_dir}/testdata/column.pkl', 'rb'))
        modelclass_replacement = pickle.load(open(f'{temp_dir}/testdata/dict.pkl', 'rb'))
        print('ion mode: negative')

    elif ionmode == 'positive':   
        model = load_model(f'{temp_dir}/testdata/model')
        model_column = pickle.load(open(f'{temp_dir}/testdata/column.pkl', 'rb'))
        modelclass_replacement = pickle.load(open(f'{temp_dir}/testdata/dict.pkl', 'rb'))
        print('ion mode: positive')

    elif ionmode == 'test':
        model = load_model(f'{temp_dir}/testdata/model')
        model_column = pickle.load(open(f'{temp_dir}/testdata/column.pkl', 'rb'))
        modelclass_replacement = pickle.load(open(f'{temp_dir}/testdata/dict.pkl', 'rb'))
        print('ion mode: test')

    else:
        print('Please select ion mode: negative or positive')

    return model, model_column, modelclass_replacement

def __apply_replacement(x,y):
    return y.get(x, x)

def ___create_pred_class(row, percent=1):
    classes = [f"{col}:{round(value * 100, 1)}%" for col, value in sorted(row.items(), key=lambda x: x[1], reverse=True) if value >= percent/100]
    return ','.join(classes)

def __pred_class(df, df_data, modelclass_replacement, model_column, model, percent=1):
    
    df_data.columns = df_data.columns.astype(str)
    X1_test = df_data[model_column].values
    y_pred_test = model.predict(X1_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)
    
    replacement_reversed = {value: key for key, value in modelclass_replacement.items()}
    y_pred_test_max_ = np.vectorize(__apply_replacement)(y_pred_test_max, replacement_reversed)

    df_class_num = pd.DataFrame({'class': list(modelclass_replacement.keys()), 'num': list(modelclass_replacement.values())})
    df_test_predclass = pd.DataFrame(y_pred_test).rename(columns=df_class_num['class'])

    create_pred_class_partial = partial(___create_pred_class, percent=percent)

    df_test_predclass['predict_1class'] = y_pred_test_max_
    df_test_predclass['predict_candidateclass'] = df_test_predclass.drop('predict_1class', axis=1).apply(create_pred_class_partial, axis=1)

    df_pred_result = df[['ms2spectrum', 'precursorion']].reset_index(drop=True).\
        merge(df_test_predclass[['predict_1class', 'predict_candidateclass']], left_index=True, right_index=True)

    return df_pred_result

def from_data_pred_classs(
        ms2spectrum :Series, 
        precursorion :Series, 
        ionmode = 'negative', 
        percent=1):
    
    """
    This function is to predict lipid class from LC-MS MS2spectrum.
    ms2spectrum and precursorion must be the same length and from same order.

    Parameters:
        ms2spectrum (Series): MS/MS spectrum represented as a Pandas Series.
        precursorion: Precursor ion represented as a Pandas Series.
        ionmode (str): Ion mode, either 'negative' / 'positive' / 'test'. Default is 'negative'.
        percent (int): Percentage of limiting value. Default is 1%.
    
    This version is supporting total 97 lipid sub classes.
    """

    if 'temp_dir' in locals():
        None
    else: 
        temp_dir = __make_temp_dir()
        __import_and_unzip_zenodo_data(temp_dir)

    model, model_column, modelclass_replacement = __load_models(temp_dir, ionmode=ionmode) 
    df, df_data = __make_table(ms2spectrum, precursorion)
    df_test_predclass = __pred_class(df, df_data, modelclass_replacement, model_column, model, percent=percent)
    return df_test_predclass

def save_pred_result(
        df_test_predclass :pd.DataFrame, 
        path = 'pred_result.csv'
        ):
    
    """
    This function saves the predicted result as a csv file.
    
    Parameters:
        df_test_predclass (pd.DataFrame): Predicted result represented as a Pandas DataFrame.
            This is also the output of the from_data_pred_classs function.
        path (str): Path to save the file. Default is 'pred_result.csv'.
    """

    df_test_predclass.to_csv(path, index=False)

#答えがあるときの正答率の評価
def __candidate_search(df_test_predclass, ont):
    df_test_predclass[['correct_class']] = pd.DataFrame(ont).reset_index(drop=True)

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
        ont :Series,
        df = None
        ):
    
    """
    This function evaluates the predicted result and also canreturns result oandas dataframe.
    If you have the correct answer, you can evaluate the accuracy of the prediction by using this.
    
    The default is to print the result, but if you want to return the result as a dataframe, 
    you can fill the df parameter is something other than None.

    Parameters:
        df_test_predclass (pd.DataFrame): Predicted result represented as a Pandas DataFrame.
            This is also the output of the from_data_pred_classs function.
        ont (Series): Correct class represented as a Pandas Series.
        df (pd.DataFrame): Dataframe to return. Default is None. 
            If you want to return the result as a dataframe, 
            you can fill the df parameter is something other than None. (like, df = 'save')
    """

    list_candidate, df_test_predclass = __candidate_search(df_test_predclass, ont)

    pred_1class_eval = len(df_test_predclass[df_test_predclass['predict_candidateclass'] == df_test_predclass['correct_class']])
    pred_cand_eval = list_candidate.count('o')
    mispred = list_candidate.count('x')

    print(f'pred_1class_eval: {pred_1class_eval}')
    print(f'pred_cand_eval: {pred_cand_eval}')
    print(f'mispred: {mispred}')

    print(f'pred 1class correct ratio: {pred_1class_eval/len(df_test_predclass)}')
    print(f'pred candidate correct ratio: {pred_cand_eval/len(df_test_predclass)}')
    print(f'mispred ratio: {mispred/len(df_test_predclass)}')

    if df == None:
        return
    else:
        return df_test_predclass


__all__ = ['from_data_pred_classs', 'save_pred_result', 'prediction_summary']
