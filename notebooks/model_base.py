import pandas as pd
import numpy as np
import random
import re
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import to_categorical
from datetime import datetime
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

#特徴量の計算とデータの選択
def _calculate_average_atomic_mass(molecular_formula):
    # Define mass
    average_atomic_masses = {'H': 1.007825, 'C': 12.000000, 'O':15.994915}
    # Calculate atomic mass from molecular formula
    elements_with_counts = re.findall(r'([A-Z][a-z]*)(\d*)', molecular_formula)
    element_counts = {element[0]: int(element[1]) if element[1] else 1 for element in elements_with_counts}
    average_atomic_mass = sum(element_counts[element] * average_atomic_masses[element] for element in element_counts)
    return average_atomic_mass

def _cal_mod(averagemz):
    num = ((averagemz % _calculate_average_atomic_mass('CH2')) % _calculate_average_atomic_mass('H2')) % (_calculate_average_atomic_mass('H14') % _calculate_average_atomic_mass('CH2')) 
    return num

def cal_df_sel_column(df):
    df.loc[:, 'ID'] = df['datasetID'].astype(str)  + '_' + df['AlignmentID'].astype(str)
    df_s = df[['MSMSspectrum','Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']]
    return df_s

#MS2スペクトルをwideに配置
def _spectrum_tidy(df_s):#MS/MS column to wide
    df_exp = pd.DataFrame(columns=range(1, 1251))
    df_exp[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']] = df_s[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']]
    df_exp = df_exp.set_index(['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']).sort_index()
    for i, row in df_s.iterrows():
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
        df_exp.loc[tuple(row[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values())    
    df_exp = df_exp.fillna(0)
    return df_exp

def _neutralloss_tidy(df_s):#neutralloss from precursor
    df_neuloss = pd.DataFrame(columns=range(0, -1251, -1), index=None)
    df_neuloss[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']] = df_s[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']]
    df_neuloss = df_neuloss.set_index(['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']).sort_index()    
    for i, row in df_s.iterrows():
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
        df_neuloss.loc[tuple(row[['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']].tolist()), xy_data.keys()] = pd.Series(xy_data)/max(xy_data.values()) 
    df_neuloss = df_neuloss.fillna(0)
    return df_neuloss

def _spectrum_neutralloss_table(df_exp, df_neuloss): #bond spectrum and neuloss data
    #spectrum: >70
    collist = df_exp.columns.astype(int).tolist()
    selected_collist_ =[num for num in collist if abs(num) >= 70]
    df_exp_2 =df_exp[selected_collist_]
    #NL: >10
    collist = df_neuloss.columns.astype(int).tolist()
    selected_collist =[num for num in collist if abs(num) >= 10]
    df_neuloss_2 =df_neuloss[selected_collist]   
    df_data = pd.concat([df_exp_2, df_neuloss_2], axis=1).reset_index()   
    return df_data

def cal_wide_df(df_s):
    df_exp = _spectrum_tidy(df_s)
    df_neuloss = _neutralloss_tidy(df_s)
    df_wide = _spectrum_neutralloss_table(df_exp, df_neuloss)
    df_wide.loc[:, 'EOvalue'] = df_wide['AverageMz'].round().astype(int) % 2
    df_wide.loc[:, 'MCHvalue'] = _cal_mod(df_wide['AverageMz'])
    return df_wide

#train, test, valデータに分割
def replacement_dict_mapping(class_list):
    replacement_dict = {class_name: i for i, class_name in enumerate(class_list)}
    return replacement_dict

def train_test_split_from_df(df_wide, columnlist_path, replacement_dict_path, y_test_path):
    ## train/test/eval
    columns_to_drop = ['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']

    with open(columnlist_path, 'rb') as file:
        loaded_negcolumns = pickle.load(file)
    df_wide2 = df_wide.set_index(columns_to_drop)
    df_wide2.columns = df_wide2.columns.astype(str)
    df_wide2 = df_wide2[loaded_negcolumns[0]].reset_index()

    X = df_wide2.drop(columns=columns_to_drop).values
    y = df_wide2['Ontology']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    train_d, test_d = train_test_split(df_wide2, test_size=0.2, random_state=42)

    replacement_dict = replacement_dict_mapping(df_wide2.Ontology.drop_duplicates().tolist())
    with open(replacement_dict_path, 'wb') as file:
        pickle.dump(replacement_dict, file)

    y_test.to_csv(y_test_path)

    return df_wide2, X_train, X_test, X_eval, y_train, y_test, y_eval, train_d, test_d, y, replacement_dict


def replace_y_number(y, y_train, y_test, y_eval, replacement_dict):
    y_train_replaced0 = y_train.replace(replacement_dict).to_numpy().astype('int64')
    y_test_replaced0 = y_test.replace(replacement_dict).to_numpy().astype('int64')
    y_eval_replaced0 = y_eval.replace(replacement_dict).to_numpy().astype('int64')
    return y_train_replaced0, y_test_replaced0, y_eval_replaced0

#onehotencodingで置換
def replace_y_onthot(y, y_train, y_test, y_eval, replacement_dict):
    y_train_replaced0, y_test_replaced0, y_eval_replaced0 =\
        replace_y_number(y, y_train, y_test, y_eval, replacement_dict)

    y_train_onehot = to_categorical(y_train_replaced0, len(y.unique()))
    y_test_onehot = to_categorical(y_test_replaced0, len(y.unique()))
    y_eval_onehot = to_categorical(y_eval_replaced0, len(y.unique()))

    return y_train_onehot, y_test_onehot, y_eval_onehot

#model
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

#param tune
def prep_param_tune(mode, X_train, y_train_onehot, X_eval, y_eval_onehot):

    if mode == 'pos':
        build_model = pos_build_model
    elif mode == 'neg':
        build_model = neg_build_model

    tuner = RandomSearch(
        build_model,
        objective='val_acc',
        max_trials=100,
        overwrite=True)

    callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=3)]
    tuner.search(X_train, y_train_onehot, validation_data=(X_eval, y_eval_onehot), callbacks=callbacks, epochs=100)

    best_hp = tuner.get_best_hyperparameters()[0]
    model = build_model(best_hp)

    best_hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.1)

    early_stopping = EarlyStopping(patience=10, verbose=0) 

    # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    return model, early_stopping, tensorboard_callback

#shap
def replace_values(x):
    if isinstance(x, str) and ('EOvalue' in x or 'MCHvalue' in x):
        return x
    else:
        num_x = int(x)
        if num_x <= 0:
            return f"NL: {-num_x}"
        else:
            return f"m/z: {num_x}"
            
def cal_df_feature(df_wide, output_path):
    df_feature = pd.DataFrame(df_wide.drop(['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID'],\
                                        axis=1).columns).reset_index().rename(columns = {'index':'id',0:'feature'})
    df_feature['featurename'] = df_feature['feature'].apply(replace_values)

    df_feature['id'] = df_feature['id'].apply(lambda x: 'Feature ' + str(x))
    df_feature.to_csv(output_path, index=False)

    return df_feature

def cal_shap_values(class_list, train_d, test_d):
    train_ds = train_d[train_d['Ontology'].isin(class_list)]
    train_ds = train_ds.drop_duplicates(subset='Metabolitename', keep='first')
    train_ds = train_ds.groupby('Ontology').head(10) 

    test_ds = test_d[test_d['Ontology'].isin(class_list)]
    test_ds = test_ds.drop_duplicates(subset='Metabolitename', keep='first')
    test_ds = test_ds.groupby('Ontology').head(10) 

    unique_ontologies = train_ds['Ontology'].unique()
    df_shap_df1 = pd.DataFrame()
    df_shap_df2 = pd.DataFrame()

    columns_to_drop = ['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']
    for ontology in unique_ontologies:
        ontology_data_train = train_ds[train_ds['Ontology'] == ontology]
        ontology_data_test = test_ds[test_ds['Ontology'] == ontology]
        df_shap_df1 = pd.concat([df_shap_df1, ontology_data_train])
        df_shap_df2 = pd.concat([df_shap_df2, ontology_data_test])

        X_train_shap = df_shap_df1.drop(columns=columns_to_drop).astype(float).values
        y_train_shap = df_shap_df1['Ontology']
        X_test_shap = df_shap_df2.drop(columns=columns_to_drop).astype(float).values
        y_test_shap = df_shap_df2['Ontology']

    return X_train_shap, y_train_shap, X_test_shap, y_test_shap

def cal_shap_results(input_path, df_feature, class_list, y_test_shap, save_excle_path):

    shap_values = np.load(input_path)

    total_shap_values = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

    total_shap_values2 = pd.DataFrame(total_shap_values).set_index(y_test_shap.reset_index(drop=False).Ontology).reset_index()
    total_shap_values2['Ontology'] = total_shap_values2['Ontology'].apply(lambda x: x if x in class_list else 'Others')
    total_shap_values2 = total_shap_values2.groupby('Ontology').mean()

    mean_shap_values2 = pd.DataFrame(mean_shap_values).set_index(y_test_shap.reset_index(drop=False).Ontology).reset_index()
    mean_shap_values2['Ontology'] = mean_shap_values2['Ontology'].apply(lambda x: x if x in class_list else 'Others')
    mean_shap_values2 = mean_shap_values2.groupby('Ontology').mean()

    #ここまでの計算結果をExcelに保存する
    with pd.ExcelWriter(save_excle_path, engine='openpyxl') as writer:
        pd.DataFrame(mean_shap_values2).transpose().to_excel(writer, sheet_name='shap_mean')
        pd.DataFrame(total_shap_values2).transpose().to_excel(writer, sheet_name='shap_total')

    mean_shap_values2.columns = feature_names
    total_shap_values2.columns = feature_names 
    feature_names = df_feature['featurename']

    # 上位20個の特徴量を抽出
    top_features_indices = pd.DataFrame(pd.DataFrame(total_shap_values2).sum(axis=0)).rename(columns={0:'sumvalue'}).sort_values("sumvalue", ascending=False)[:20].reset_index()['index'].tolist()
    top_feature_names = feature_names.iloc[top_features_indices]
    top_feature_df = pd.DataFrame(mean_shap_values2)[top_features_indices].T
    top_feature_df.index = top_feature_names

    return top_feature_df

def plot_shap_results(top_feature_df, cmap, custom_legend_order, save_path, title='SHAP summary plot'):
    top_feature_df2 = top_feature_df.T
    top_feature_df2 = top_feature_df2.loc[custom_legend_order]

    font = {'family': 'Arial', 'weight': 'normal', 'size': 12}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(len(top_feature_df2)):
        ax.barh(top_feature_df2.columns, top_feature_df2.iloc[i], left=top_feature_df2.iloc[:i].sum(), color=cmap[i], ec='black', linewidth=1)
    plt.title(title, fontsize=18)
    plt.xlabel('mean (|SHAP value|)', fontsize=14)
    plt.ylabel('')
    plt.tick_params(labelsize=12)
    plt.legend(custom_legend_order, title='Major 10 lipid class', fontsize=14, title_fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.edgecolor"] = 'black'

    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()

#result summary
def plot_learning_curve(history, save_path, title = 'Learning curves'): 
    plt.figure(figsize=(8, 5))

    # 学習過程の可視化
    plt.plot(history.epoch, history.history['loss'], label='train')
    plt.plot(history.epoch, history.history['val_loss'], label='valid')
    plt.xticks(range(0, len(history.epoch)+1, 50))
    plt.ylabel('Loss', fontsize=20)  
    plt.xlabel('Epochs', fontsize=20)  
    plt.title(title, fontsize=20) 
    plt.legend(loc='upper right', fontsize=16)  
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()

def modelpred_summary_acc(model, X_train, y_train_onehot, X_test, y_test_onehot):
    y_pred_train = model.predict(X_train)
    y_pred_train_max = np.argmax(y_pred_train, axis=1)
    _, y_train_acc = np.where(y_train_onehot > 0) 

    y_pred_test = model.predict(X_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)
    _, y_test_acc = np.where(y_test_onehot > 0) 

    # 正答率
    train_accuracy = accuracy_score(y_train_acc, y_pred_train_max)
    test_accuracy = accuracy_score(y_test_acc, y_pred_test_max)

    print('Neural Network_train :', train_accuracy)
    print('Neural Network_test :', test_accuracy)

    return y_pred_train, y_pred_train_max, y_train_acc, y_pred_test, y_pred_test_max, y_test_acc

def _check_ontology_in_pred_class(row):
    ontology = row['Ontology']
    pred_class = row['pred_class']
    return ontology in pred_class

def _create_pred_class(row):
    classes = [f"{col}:{round(value, 3)}" for col, value in sorted(row.items(), key=lambda x: x[1], reverse=True) if value >= 0.01]
    return ','.join(classes)

def modelpred_summary_detail(df_wide, replacement_dict_reversed, y_data, y_pred, y_pred_max, y_acc, save_path):

    columns_to_drop = ['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']
    df_predclass = pd.DataFrame(y_pred).rename(columns=replacement_dict_reversed)

    df_predclass['pred_1class'] = np.vectorize(replacement_dict_reversed.get)(y_pred_max)
    df_predclass['pred_class'] = df_predclass.drop('pred_1class', axis=1).apply(_create_pred_class, axis=1)
    df_predclass['pred_number'] = df_predclass['pred_class'].apply(lambda x: len(x.split(',')))

    df_predclass_v = df_predclass[['pred_1class','pred_class','pred_number']]
    df_copy = df_wide.drop(columns=columns_to_drop).reset_index().rename(columns={'index':'id'})['id'].copy(deep=True)
    df_predclass_v = df_predclass_v.assign(id=df_copy)

    df_predclass_v['id'] = y_data.index.tolist()
    df_predclass_v = df_predclass_v.set_index('id')

    df_predclass.index = y_data.index.tolist()
    df_predclass.to_csv(save_path)

    df_mispred = pd.concat([df_predclass_v, y_data], axis=1)[y_acc != y_pred_max]
        
    df_mispred['TF'] = df_mispred.apply(_check_ontology_in_pred_class, axis=1)

    pred1st =  len(df_predclass_v[y_acc == y_pred_max]) / len(df_predclass)
    predcandidate = (len(df_predclass_v[y_acc == y_pred_max]) + len(df_mispred.query('TF == True'))) / len(df_predclass)
    pred1st_num = len(df_predclass_v[y_acc == y_pred_max])
    candidate_num = len(df_predclass_v[y_acc == y_pred_max]) + len(df_mispred.query('TF == True'))
    all_num = len(df_predclass)

    print('1st pred :', pred1st)
    print('candidate :', predcandidate)

    print('1st pred num:', pred1st_num)
    print('candidate num:',candidate_num)

    print('all num:', all_num)

    return pred1st_num, candidate_num, all_num, df_predclass_v

def plot_bar_plot(pred1st_num, candidate_num, all_num, save_path):
    # データの準備
    test_pred_1st = round(pred1st_num/all_num * 100, 1)
    test_pred_candidate = round(candidate_num/all_num * 100, 1)
    percentages = [test_pred_1st, test_pred_candidate-test_pred_1st, 100-test_pred_candidate]
    percentages2 = [f'{test_pred_1st}%','','']#round(test_pred_candidate-test_pred_1st, 1), ]
    categories = ["pred_1st", "pred_candidate", "nonpredict"]

    fig = plt.figure(figsize=(8, 0.6))  # figsizeを調整して適切なサイズに設定
    bar_width = 0.6

    # バーの中心位置を計算
    bar_centers = [ percentages[0]/2, percentages[0] + percentages[1]/2, percentages[0] + percentages[1] + percentages[2]/2,]
    cl = plt.cm.tab20.colors
    deep_palette = [cl[18], cl[13], cl[0]]

    # 水平な積み上げ棒グラフを描画
    bars = plt.barh(1, percentages[0], height=bar_width, color=deep_palette[1], label='pred_1st')
    plt.barh(1, percentages[1], left=percentages[0], height=bar_width, color=deep_palette[0], label='pred_candidate')
    plt.barh(1, percentages[2], left=percentages[0] + percentages[1], height=bar_width, color=deep_palette[2], label='nonpredict')
    # パーセントラベルを描画
    for category, center, percentage in zip(categories, bar_centers, percentages2):
        plt.text(center+0.5, 1, percentage, ha='center', va='center', fontsize=11, color='black')
        
    # ラベルを設定
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.yticks([])
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    plt.xlim(0, 100) 
    plt.title('Test result', fontsize=18)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()

#canopusとの比較用データ
def conv_subclass(test_df_predclass, y_test, ont_df_path, save_path, subclasslist):
    df_allexp = pd.concat([test_df_predclass, y_test], axis=1)
    df_allexp['TF'] = df_allexp.apply(_check_ontology_in_pred_class, axis=1)
    select_ont = pd.read_csv(ont_df_path)

    df_allexp2 = pd.merge(df_allexp.reset_index(), select_ont, on='Ontology', how='inner').rename(columns={'subclass':'cor_subclass'})
    df_allexp2 = pd.merge(df_allexp2, select_ont, left_on = 'pred_1class', right_on = 'Ontology', how='inner').rename(columns={'subclass':'pred_subclass','Ontology_y':'pred_ont'})

    df_subclass_pred = pd.DataFrame(columns=['Subclass', 'Correct_Predictions', 'Incorrect_Predictions', 'Total'])

    for i in subclasslist:
        df_allexp2_i = df_allexp2[df_allexp2['cor_subclass'] == i]
        correct_predictions = len(df_allexp2_i[df_allexp2_i['cor_subclass'] == df_allexp2_i['pred_subclass']])
        incorrect_predictions = len(df_allexp2_i) - correct_predictions
        total_samples = len(df_allexp2_i)

        new_row = {'Subclass': i,
                'Correct_Predictions': correct_predictions,
                'Incorrect_Predictions': incorrect_predictions,
                'Total': total_samples}
        
        # ループ内でデータを追加
        df_subclass_pred = pd.concat([df_subclass_pred, pd.DataFrame([new_row])], ignore_index=True)

    # ループが終了した後でCSVファイルに保存
    df_subclass_pred.to_csv(save_path, index=False)

#rendom forestのparameter tune
def rf_random_search(X_train, y_train_replaced0, X_test, y_test_replaced0):
    RFC_random_grid = {RandomForestClassifier(): {"n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400],
                                                "criterion": ["gini", "entropy"],
                                                "max_depth": [1, 5, 10, 15, 20, 25, 30, 35, 40, None],
                                                }}

    max_score = 0
    # ランダムフォレストの実行
    for model, param in tqdm(RFC_random_grid.items()):
        clf = RandomizedSearchCV(model, param, n_iter=10)
        clf.fit(X_train, y_train_replaced0)
        pred_y = clf.predict(X_test)
        score = accuracy_score(y_test_replaced0, pred_y)

        if max_score < score:
            max_score = score
            best_param = clf.best_params_

    print('best score')
    print("parameter:{}".format(best_param))

    rf_y_pred = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test_replaced0, rf_y_pred)
    rf_precision = precision_score(y_test_replaced0, rf_y_pred, average='macro')
    rf_recall = recall_score(y_test_replaced0, rf_y_pred, average='macro')
    rf_f1 = f1_score(y_test_replaced0, rf_y_pred, average='macro')

    print(rf_accuracy, rf_precision, rf_recall, rf_f1)

    return rf_accuracy, rf_precision, rf_recall, rf_f1

#svmのparameter tune
def svm_random_search(X_train, y_train_replaced0, X_test, y_test_replaced0):
    SVC_random_grid = {SVC(): {"C": [0.0, 0,5, 1.0],  
                                "kernel": ["linear", "rbf", "sigmoid"],
                                "decision_function_shape": ["ovo", "ovr"]}}

    # ランダムサーチ
    max_score = 0
    for model, param in SVC_random_grid.items():
        clf = RandomizedSearchCV(model, param_distributions=param, n_iter=100)
        clf.fit(X_train, y_train_replaced0)
        pred_y = clf.predict(X_test)
        score = accuracy_score(y_test_replaced0, pred_y)

        if max_score < score:
            max_score = score
            best_param = clf.best_params_

    print('best score')
    print("parameter:{}".format(best_param))

    # 最適なモデルで予測と評価
    svm_y_pred = clf.predict(X_test)
    svm_accuracy = accuracy_score(y_test_replaced0, svm_y_pred)
    svm_precision = precision_score(y_test_replaced0, svm_y_pred, average='macro') 
    svm_recall = recall_score(y_test_replaced0, svm_y_pred, average='macro')
    svm_f1 = f1_score(y_test_replaced0, svm_y_pred, average='macro')

    print(svm_accuracy, svm_precision, svm_recall, svm_f1)

    return 

#knnのparameter tune
def knn_random_search(X_train, y_train_replaced0, X_test, y_test_replaced0):
    knn_random_grid = {KNeighborsClassifier(): {'n_neighbors': [1,5,10,15,20],  # 1から20の整数乱数
                                                'weights': ['uniform', 'distance'],
                                                'p': [1, 2]}}

    # ランダムサーチ
    max_score = 0
    for model, param in knn_random_grid.items():
        clf = RandomizedSearchCV(model, param_distributions=param, n_iter=100)
        clf.fit(X_train, y_train_replaced0)
        pred_y = clf.predict(X_test)
        score = accuracy_score(y_test_replaced0, pred_y)

        if max_score < score:
            max_score = score
            best_param = clf.best_params_

    print('best score')
    print("parameter:{}".format(best_param))

    # 最適なモデルで予測と評価
    knn_y_pred = clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test_replaced0, knn_y_pred)
    knn_precision = precision_score(y_test_replaced0, knn_y_pred, average='macro')
    knn_recall = recall_score(y_test_replaced0, knn_y_pred, average='macro')
    knn_f1 = f1_score(y_test_replaced0, knn_y_pred, average='macro')

    print(knn_accuracy, knn_precision, knn_recall, knn_f1)

    return knn_accuracy, knn_precision, knn_recall, knn_f1

#nnにおけるscore summary
def nn_score_summary(model, X_test, y_test_onehot):
    y_pred_test = model.predict(X_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)
    _, y_test_acc = np.where(y_test_onehot > 0) 

    nn_accuracy = accuracy_score(y_test_acc, y_pred_test_max)
    nn_precision = precision_score(y_test_acc, y_pred_test_max, average='macro')
    nn_recall = recall_score(y_test_acc, y_pred_test_max, average='macro')
    nn_f1 = f1_score(y_test_acc, y_pred_test_max, average='macro')

    print(nn_accuracy, nn_precision, nn_recall, nn_f1)

    return nn_accuracy, nn_precision, nn_recall, nn_f1


