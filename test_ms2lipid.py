import pandas as pd
from src.ms2lipid.ms2slipid_func import from_data_pred_classs, save_pred_result, prediction_summary


df_test = pd.read_csv('tests/testdata.csv' )
ms2 = df_test['MSMSspectrum']
prec = df_test['AverageMz']
ont = df_test['Ontology']


#from_data_pred_classs
def test_from_data_pred_classs_type():
    assert callable(from_data_pred_classs)

def test_from_data_pred_classs_value():
    result = from_data_pred_classs(ms2, prec, ionmode='test')
    assert isinstance(result, pd.DataFrame)


#save_pred_result
def test_save_pred_result_type():
    assert callable(save_pred_result)

def test_save_pred_result_value():
    result = from_data_pred_classs(ms2, prec, ionmode='test')
    
    save_pred_result(result, path='pred_result.csv')
    loaded_result = pd.read_csv('pred_result.csv')
    assert result.equals(loaded_result)


#prediction_summary
def test_prediction_summary_type():
    assert callable(prediction_summary)

def test_prediction_summary_value():
    result = from_data_pred_classs(ms2, prec, ionmode='test')
    prediction_summary(result, ont)
 
def test_prediction_summary_type2():
    result = from_data_pred_classs(ms2, prec, ionmode='test')
    result2 = prediction_summary(result, ont, 'x')
    assert isinstance(result2, pd.DataFrame)

def test_prediction_summary_value2():
    result = from_data_pred_classs(ms2, prec, ionmode='test')
    result2 = prediction_summary(result, ont, 'x')

    save_pred_result(result2, path='pred_result2.csv')
    loaded_result2 = pd.read_csv('pred_result2.csv')
    assert result2.equals(loaded_result2)
