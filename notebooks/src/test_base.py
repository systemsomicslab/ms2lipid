import zipfile
import os
import re
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import pickle 

###canopus
#ms fileの作成
def make_ms_file(df_select, output_file_path):
    df_select = df_select.reset_index(drop=True)
    invalid_chars = r'[\\/:*?"<>|]'

    for i in range(len(df_select)):
        #compound = df_select['Metabolitename'][i]
        formula = df_select['Formula'][i]
        parentmass = df_select['AverageMz'][i]
        ionization = df_select['Adducttype'][i]
        
        ms2_data = df_select['MSMSspectrum'][i]
        elements = ms2_data.split()  
        id = df_select.index.tolist()[i]
        metabolite = re.sub(invalid_chars, '_', df_select['Metabolitename'][i])
        ontology = re.sub(invalid_chars, '_', df_select['Ontology'][i])
        name_ont = f"{id}_{metabolite}_{ontology}"

        ms2_split_data = []
        for element in elements:
            parts = element.split(':')
            ms2_split_data.append(f"{parts[0]} {parts[1]}")

        output_file = f"{output_file_path}/{name_ont}.ms"
        with open(output_file, 'w', newline='') as file:
            
            file.write(f">compound {metabolite}_{ontology}\n")
            file.write(f">formula {formula}\n")
            file.write(f">parentmass {parentmass}\n")
            file.write(f">ionization {ionization}\n")
            
            file.write(">ms2\n")
            file.write('\n'.join(ms2_split_data))

def make_zipfile(folder_path, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.ms'):
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

def read_canopus_output(canopus_result_path, select_ont_path, ytest_path, model_test_result, df_wide, df_2):
    df_ytest = pd.read_csv(ytest_path).rename(columns={'Unnamed: 0':'index_ID'})
    select_ont = pd.read_csv(select_ont_path)
    df_ytest = df_ytest[df_ytest['Ontology'].isin(select_ont.Ontology.tolist())].merge(select_ont, how='left', on='Ontology')

    df_wide2 = df_wide.reset_index().merge(df_ytest, left_on=['index', 'Ontology'], right_on = ['index_ID', 'Ontology'], how='inner').drop(columns='index_ID')
    df_wide2 = df_2[['ID']].reset_index().rename(columns={'index':'index_id'}).merge(df_wide2, on='ID', how='inner')

    df_canopus = pd.read_csv(canopus_result_path, sep="\t")
    df_canopus['index_id'] = df_canopus['id'].apply(lambda x: x.split("_")[1])
    df_canopus_ = df_canopus[['id','ClassyFire#subclass','ClassyFire#level 5','index_id']].rename(columns={'ClassyFire#subclass':'canopus_pred'})
    df_canopus_['index_id'] = df_canopus_['index_id'].astype(int)

    df_test_result = pd.read_csv(model_test_result)
    df_test_result = df_test_result[['index', 'cor_subclass', 'pred_subclass']].rename(columns={'cor_subclass':'subclass', 'pred_subclass':'model_pred'})

    df_model_canopus = df_wide2.merge(df_canopus_, how='left', on='index_id').\
        merge(df_test_result, on=['subclass', 'index'], how='inner')[['ID', 'subclass', 'canopus_pred', 'model_pred']]

    return df_model_canopus

def df_comp_pred_result(df):

    df_cano_class_summary = pd.DataFrame(columns=['Subclass','total', 'model_correct','model_mis', 'canopus_correct','canopus_mis','canopus_nan'])
    subclasslist = df['subclass'].unique()

    for i in subclasslist:
        dfi = df[df['subclass'] == i]
        total = len(dfi)
        model_correct = len(dfi[dfi['model_pred'] == i])
        model_mis = len(dfi[dfi['model_pred'] != i])
        canopus_correct = len(dfi[dfi['canopus_pred'] == i])
        canopus_nan = len(dfi[dfi['canopus_pred'].isna()])
        canopus_mis = total - canopus_correct - canopus_nan

        new_row = {'Subclass': i,
                'total':total,
                'model_correct': model_correct,
                'model_mis': model_mis,
                'canopus_correct': canopus_correct,
                'canopus_mis':canopus_mis,
                'canopus_nan':canopus_nan}

        df_cano_class_summary = pd.concat([df_cano_class_summary, pd.DataFrame([new_row])], ignore_index=True)

    return df_cano_class_summary

def make_classification_reports(df_comp, excel_file_path):
    canopus_report = classification_report(df_comp['subclass'], df_comp['canopus_pred'], output_dict=True, zero_division=0)
    model_report = classification_report(df_comp['subclass'], df_comp['model_pred'], output_dict=True, zero_division=0)

    excel_file = excel_file_path
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        pd.DataFrame(canopus_report).transpose().to_excel(writer, sheet_name='Canopus Report')
        pd.DataFrame(model_report).transpose().to_excel(writer, sheet_name='Model Report')

def plot_canopus_result(df_cano_class_pred, save_fig_path, title='Percentage of correct answers in subclass'):

    df_all = df_cano_class_pred.rename(columns={'Subclass':'class'}).set_index('class')

    df_all_2 = pd.DataFrame()
    df_all_2['model_correct'] = (df_all['model_correct']/df_all['total'])*100
    df_all_2['model_mispred'] = 100-df_all_2['model_correct']

    df_all_2['canopus_correct'] = (df_all['canopus_correct']/df_all['total'])*100
    df_all_2['canopus_mispred'] = (df_all['canopus_mis']/df_all['total'])*100
    df_all_2['canopus_nonpred'] = (df_all['canopus_nan']/df_all['total'])*100

    df_all2 = df_all_2.reset_index().sort_values('class')

    # Create a bar chart for model data
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35  # Width of each bar

    y = np.arange(len(df_all2['class']))

    red = '#d50310'
    blue = '#007cb6'
    black = '#2f2f2f'
    custom_colors = [red, blue, black]

    # Plot model_1st_correct and model_mispredict bars
    ax.barh(y - width/2, df_all2['model_correct'], width, label='model_correct', color=custom_colors[0])
    ax.barh(y - width/2, df_all2['model_mispred'], width, label='model_mispred', color=custom_colors[1], left=df_all2['model_correct'])
    ax.barh(y + width/2, df_all2['canopus_correct'], width, label='canopus_correct', color=custom_colors[0])
    ax.barh(y + width/2, df_all2['canopus_mispred'], width, label='canopus_mispred', color=custom_colors[1], left=df_all2['canopus_correct'])
    ax.barh(y + width/2, df_all2['canopus_nonpred'], width, label='canopus_nonpred', color=custom_colors[2], left=df_all2['canopus_correct'] + df_all2['canopus_mispred'])

    #ax.set_ylabel('Class', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(df_all2['class'], fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))

    bars1= df_all2[['model_correct','model_mispred']]
    bars2= df_all2[['canopus_correct','canopus_mispred','canopus_nonpred']]

    for n in bars1:
        for i, (cs, ab, pc, sn) in enumerate(zip(bars1.iloc[:, 0:].cumsum(1)[n],
                                            bars1[n], bars1[n],df_all.sort_index()['total'])):
            if pc > 5.0:  
                plt.text(cs - ab / 2 + 0.5, i - width/2 + 0.04, str(np.round(pc, 1)) + '%',
                        va='center', ha='center', rotation=0, fontsize=8, c='white')
            plt.text(100 + 2, i, f'n={sn}', va='center', ha='left', fontsize=10, alpha=0.8, c='black')

    for n in bars2:
        for i, (cs, ab, pc) in enumerate(zip(bars2.iloc[:, 0:].cumsum(1)[n],
                                            bars2[n], bars2[n])):
            if pc > 5.0:  
                plt.text(cs - ab / 2 + 0.5, i + width/2 + 0.04, str(np.round(pc, 1)) + '%',
                        va='center', ha='center', rotation=0, fontsize=8, c='white')
                

    for i in range(len(df_all2)):
        plt.axhline(i, color='black', linewidth=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0., ncol = 1)
    plt.yticks(size = 9)
    plt.tight_layout()
    plt.savefig(save_fig_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()

###riken lipidomics
def replacement_dict_mapping(class_list):
    replacement_dict = {class_name: i for i, class_name in enumerate(class_list)}
    return replacement_dict

def apply_replacement(x,y):
    return y.get(x, x)

def create_pred_class(row):
    classes = [f"{col}:{round(value, 3)}" for col, value in sorted(row.items(), key=lambda x: x[1], reverse=True) if value >= 0.01]
    return ','.join(classes)

def check_ontology_in_pred_class(row):
    ontology = row['Ontology']
    pred_class = row['pred_class']
    return ontology in pred_class

def load_models(df_wide, replacement_dict_path, column_path, model_path):
    #データ読み込み
    loaded_model = load_model(model_path)
    with open(column_path, 'rb') as file:
        loaded_columns = pickle.load(file)
    with open(replacement_dict_path, 'rb') as file:
        replacement_dict = pickle.load(file)

    return replacement_dict, loaded_columns, loaded_model

def pred_data_adj(df_wide, replacement_dict, loaded_columns):

    replacement_dict_reversed = {value: key for key, value in replacement_dict.items()}

    replacement_dict_reversed_df = pd.DataFrame(columns=['replaced_name', 'name'])
    for replacement, original in replacement_dict_reversed.items():
        replacement_dict_reversed_df = pd.concat([replacement_dict_reversed_df, pd.DataFrame({'replaced_name': [replacement], 'name': [original]})], ignore_index=True)


    columns_to_drop = ['Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID']
    df_wide2 = df_wide.set_index(columns_to_drop)
    df_wide2.columns = df_wide2.columns.astype(str)
    df_wide2 = df_wide2[loaded_columns[0]].reset_index()

    return df_wide2, columns_to_drop, replacement_dict_reversed, replacement_dict_reversed_df

def pred_data_by_model(df_wide2, replacement_dict, replacement_dict_reversed, replacement_dict_reversed_df, loaded_model, columns_to_drop):
    X1_test = df_wide2.drop(columns=columns_to_drop).values
    y1_test = df_wide2['Ontology']
    y1_test_replaced = y1_test.replace(replacement_dict)

    replacement_dict_num = replacement_dict_mapping(y1_test_replaced.drop_duplicates().tolist())
    replacement_dict_reversed_num = {value: key for key, value in replacement_dict_num.items()}

    replacement_dict_reversed_num_df = pd.DataFrame(columns=['re_replaced_name', 'replaced_name'])
    for replacement, original in replacement_dict_reversed_num.items():
        replacement_dict_reversed_num_df = pd.concat([replacement_dict_reversed_num_df, pd.DataFrame({'re_replaced_name': [replacement], 'replaced_name': [original]})], ignore_index=True)


    y1_test_replaced_num = y1_test_replaced.replace(replacement_dict_num)
    y1_test_onehot = to_categorical(y1_test_replaced_num, len(y1_test_replaced_num.drop_duplicates()))

    y_pred_test = loaded_model.predict(X1_test)
    y_pred_test_max = np.argmax(y_pred_test, axis=1)
    _, y_test_acc = np.where(y1_test_onehot > 0)

    y_pred_test_max_ = np.vectorize(apply_replacement)(y_pred_test_max, replacement_dict_reversed)
    y_test_acc_ = np.vectorize(apply_replacement)(np.vectorize(apply_replacement)(y_test_acc, replacement_dict_reversed_num), replacement_dict_reversed)

    df_ontname = pd.merge(replacement_dict_reversed_num_df, replacement_dict_reversed_df, on='replaced_name', how='right')   
    df_test_predclass = pd.DataFrame(y_pred_test).rename(columns=df_ontname['name'])

    df_test_predclass['pred_1class'] = y_pred_test_max_
    df_test_predclass['pred_class'] = df_test_predclass.drop('pred_1class', axis=1).apply(create_pred_class, axis=1)
    df_test_predclass['pred_number'] = df_test_predclass['pred_class'].apply(lambda x: len(x.split(',')))

    df_copy = df_test_predclass.reset_index().rename(columns={'index':'idx'})['idx'].copy(deep=True)
    df_test_predclass = df_test_predclass.assign(idx=df_copy)
    df_test_predclass['Ontology'] = y_test_acc_
        
    df_test_predclass_v = df_test_predclass[['pred_1class','pred_class','pred_number']]
    df_copy = df_wide2.reset_index().rename(columns={'index':'idx'})['idx'].copy(deep=True)
    df_test_predclass_v = df_test_predclass_v.assign(idx=df_copy)

    return df_test_predclass, df_test_predclass_v, df_wide2, y_test_acc_, y_pred_test_max_

def cal_accscore(df_wide, replacement_dict_path, column_path, model_path):
    #データ読み込み
    replacement_dict, loaded_columns, loaded_model =\
          load_models(df_wide, replacement_dict_path, column_path, model_path)

    df_wide2, columns_to_drop, replacement_dict_reversed, replacement_dict_reversed_df =\
          pred_data_adj(df_wide, replacement_dict, loaded_columns)

    df_wide3 = df_wide2[df_wide2['Ontology'].isin(replacement_dict_reversed_df.name.unique())]
    
    df_test_predclass, df_test_predclass_v, df_wide3, y_test_acc_, y_pred_test_max_ =\
          pred_data_by_model(df_wide3, replacement_dict, replacement_dict_reversed, replacement_dict_reversed_df, loaded_model, columns_to_drop)
        
    df_mispred = pd.concat([df_test_predclass_v.set_index('idx')[y_test_acc_ != y_pred_test_max_].reset_index(drop=True),\
                pd.DataFrame(df_wide3[y_test_acc_ != y_pred_test_max_]).reset_index().rename(columns={'index':'id'})],axis=1)
        
    df_mispred['TF'] = df_mispred.apply(check_ontology_in_pred_class, axis=1)
        
    pred1st =  len(df_wide3[y_test_acc_ == y_pred_test_max_]) 
    predcandidate = (len(df_wide3[y_test_acc_ == y_pred_test_max_]) + len(df_mispred.query('TF == True')))
    num = len(df_test_predclass)

    return pred1st, predcandidate, num, df_test_predclass

def make_pred_summary(df_wide, replacement_dict_path, column_path, model_path, datasetid, type='isin',
                      title = 'test Accuracy'):
    
    df_wide['id'] = df_wide['ID'].str.split('_').str[0].astype(int)
    if type == 'btw':
        dataset_select = df_wide[df_wide['id'].between(datasetid[0], datasetid[1])]
    else:
        dataset_select = df_wide[df_wide['id'].isin(datasetid)]

    print(title)
    pred1st, predcandidate, num, df_test_predclass = cal_accscore(dataset_select, replacement_dict_path, column_path, model_path)
    print('1st pred :', pred1st)
    print('candidate :', predcandidate)
    print('total:', num)
    print('----')

    return pred1st, predcandidate, num, df_test_predclass

###human cohort
def make_pred_class_list_column(df_pred):
    pred_class_list = []
    for index, row in df_pred.iterrows():
        split_by_commas = row['pred_class'].split(',')
        elements_before_colon = []
        for item in split_by_commas:
            if ':' in item:
                elements_before_colon.append(item.split(':')[0])
            else:
                elements_before_colon.append(item)
        pred_class_list.append(elements_before_colon)
    return pred_class_list

def check_ontology_in_pred_class(row):
    ontology = row['Ontology']  
    pred_class_list = row['pred_class_list']  
    for item in pred_class_list:
        for sub_item in item.split(','):
            if ontology == sub_item.replace(' ',''):
                return True
    return False

def extract_x(row):
    return row['ID'].split('_')[1] if '_' in row['ID'] else None
    
    
def plot_spectrum(df_wide, Alignmnentid):

    Alignmnentid = 8864
    df_wide.index = df_wide.apply(extract_x, axis=1)
    columns_to_drop = [ 'Metabolitename','Ontology', 'AverageMz', 'Adducttype','ID', 'EOvalue',	'MCHvalue']
    df_wide = df_wide.drop(columns=columns_to_drop).astype(float)
    df1= pd.DataFrame(df_wide.loc[Alignmnentid]).reset_index().rename(columns={'index':'mz',Alignmnentid:'exp'}).astype('float')

    fig, axs = plt.subplots(1, 1, figsize=(4, 3))

    axs.plot(df1['mz'], df1['exp'], color='black')
    axs.set_xlabel('M/Z', fontsize=12)
    axs.set_ylabel('Intensity', fontsize=12)
    axs.set_title('Alignment ID: ' + str(Alignmnentid), fontsize=12)
    axs.set_xlim(0, 1250)

    #plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()

def plot_ms2_peaktable(df, Alignmnentid):

    element_list = df[df['AlignmentID'] == Alignmnentid]['MSMSspectrum'].str.split(' ')

    mz_list = []
    int_list = []

    for element in element_list:
        for i in element:
            i_div = i.split(':')
            mz = i_div[0]
            int_val = i_div[1]

            mz_list.append(mz)
            int_list.append(int_val)

    df = pd.DataFrame(list(zip(mz_list, int_list))).rename(columns={0:'mz',1:'int'}).astype(float)
    
    return df
