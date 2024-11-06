import pandas as pd
import pytest
#from ms2lipid.ms2lipid import *

path_test_csv_neg = 'tests/test_data_csv_neg.csv'
path_test_csv_pos = 'tests/test_data_csv_pos.csv'
path_test_msdial_txt_neg = 'tests/test_data_msdial_txt_neg.txt'
path_test_msp_pos = 'tests/test_data_msp_pos.msp'
path_test_txt_neg = 'tests/test_data_txt_neg.txt'

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ms2lipid')))
from ms2lipid import *

#pred class
def test_predclass_calla():
    assert callable(predclass)

def test_predclass_isins():
    result = predclass(path_test_csv_neg, ionmode='negative', ms2spc_name = 'MSMSspectrum', prec_name = 'AverageMz', ID = 'sampleid')
    assert isinstance(result, pd.DataFrame)

#prediction_summary
def test_prediction_summary_calla():
    assert callable(prediction_summary)

def test_prediction_summary_isins():
    result = predclass(path_test_csv_neg, ionmode='negative', ms2spc_name = 'MSMSspectrum', prec_name = 'AverageMz', ID = 'sampleid')
    result2 = prediction_summary(result, path_test_csv_neg, class_name = 'Ontology', showdf = True)
    assert isinstance(result2, pd.DataFrame)

def test_prediction_summary_save_equals():
    result = predclass(path_test_csv_neg, ionmode='negative', ms2spc_name = 'MSMSspectrum', prec_name = 'AverageMz', ID = 'sampleid')
    result2 = prediction_summary(result, path_test_csv_neg, class_name = 'Ontology', showdf = True, exppath = 'test.csv')
    loaded_result2 = pd.read_csv('test.csv')
    assert result2.equals(loaded_result2)

#demo data
def test_csvdata():
    result = predclass(path_test_csv_pos, ionmode='positive', ms2spc_name = 'MSMSspectrum', prec_name = 'AverageMz', ID = 'sampleid')
    prediction_summary(result, path_test_csv_pos, class_name = 'Ontology', showdf = True, exppath = 'test.csv')
    df = pd.read_csv('test.csv')
    assert (df['predict_1class'] == df['correct_class']).all(), "Mismatch found in 'predict_1class' and 'correct_class' columns"

def test_mspdata():
    result = predclass(path_test_msp_pos, ionmode='positive', ms2spc_name = 'MS/MS spectrum', prec_name = 'Average Mz', ID = 'Alignment ID')
    prediction_summary(result, path_test_msp_pos, class_name = 'Ontology', showdf = True, exppath = 'test.csv')
    df = pd.read_csv('test.csv')
    assert (df['predict_1class'] == df['correct_class']).all(), "Mismatch found in 'predict_1class' and 'correct_class' columns"

def test_txtdata():
    result = predclass(path_test_txt_neg, ionmode='negative', ms2spc_name = 'MS/MS spectrum', prec_name = 'Average Mz', ID = 'Alignment ID')
    prediction_summary(result, path_test_txt_neg, class_name = 'Ontology', showdf = True, exppath = 'test.csv')
    df = pd.read_csv('test.csv')
    assert (df['predict_1class'] == df['correct_class']).all(), "Mismatch found in 'predict_1class' and 'correct_class' columns"

def test_msdialdata():
    result = predclass(path_test_msdial_txt_neg, ionmode='negative', format ='MSDIAL', ms2spc_name = 'MS/MS spectrum', prec_name = 'Average Mz', ID = 'Alignment ID')
    prediction_summary(result, path_test_msdial_txt_neg, class_name = 'Ontology',format ='MSDIAL',  showdf = True, exppath = 'test.csv')
    df = pd.read_csv('test.csv')
    assert (df['predict_1class'] == df['correct_class']).all(), "Mismatch found in 'predict_1class' and 'correct_class' columns"

if __name__ == "__main__":
    pytest.main([__file__])