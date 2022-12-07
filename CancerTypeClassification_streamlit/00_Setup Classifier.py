# Cancer Classifier by RNA-Seq Gene Expression

#### Zhanyang Zhu Zhanyang.Zhu@Gmail.com 12/3/2022 for UCSD Machine Learning Bootcamp Capstone Project 

## Function: Assign one of the following cancer types given RNA-Seq gene expression data of a patient
#### 1. BRCA: Breast invasive carcinoma
    #### 2. COAD: Colon adenocarcinoma
    #### 3. KIRC: Kidney renal clear cell carcinoma
    #### 4. LUAD: Lung adenocarcinoma
    #### 5. PRAD: Prostate adenocarcinoma
    #### 6. SKCM: Skin Cutaneous Melanoma
    #### 7. THCA: Thyroid carcinoma
    #### 8. LGG: Brain Lower Grade Glioma 

## Input Data:

#### Samples (instances) are stored column-wise. Variables (attributes in rows) of each sample are RNA-Seq gene expression levels measured by illumina HiSeq RNA-seq V2 platform.
### See data sets from https://www.synapse.org/#!Synapse:syn2812961
#### 1. unc.edu_BRCA_IlluminaHiSeq_RNASeqV2.geneExp.tsv - BRCA: Breast invasive carcinoma
#### 2. unc.edu_COAD_IlluminaHiSeq_RNASeqV2.geneExp.tsv - COAD: Colon adenocarcinoma

## Model Training: 
#### 1. https://github.com/ZhanyangZhuSD/UCSDMLCapstone/blob/main/GeneExpressionCancerRNA-Seq_FullData_XGBoost.ipynb
#### 2. https://github.com/ZhanyangZhuSD/UCSDMLCapstone/blob/main/UCSD%20ML%20Bootcamp%20Capstone%20Report%20ZhanyangZhu.pdf
import streamlit as st
import pandas as pd
import plotly.express as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost
from xgboost import XGBClassifier

import pickle
import warnings
# import time
from datetime import datetime

import os.path

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("Cancer Classifier by RNA-Seq Gene Expression")
st.sidebar.title('')

uploaded_file = st.file_uploader('Upload RNASeq Gene Expression File:', help='Select RNA-Seq gene expression data file')

cancer_description = {"BRCA": "Breast invasive carcinoma", 
              "COAD": "Colon adenocarcinoma",
              "KIRC": "Kidney renal clear cell carcinoma",
              "LUAD": "Lung adenocarcinoma",
              "PRAD": "Prostate adenocarcinoma",
              "SKCM": "Skin Cutaneous Melanoma",
              "THCA": "Thyroid carcinoma",
              "LGG": "Brain Lower Grade Glioma"
             }
st.session_state['cancer_description'] = cancer_description 
           
if uploaded_file is not None:
    st.write("loading..."+ uploaded_file.name)
    # df = pd.read_csv(uploaded_file,delimiter='\t', index_col=["gene_id"])
    unc_test = pd.read_csv(uploaded_file, delimiter='\t', index_col=["gene_id"])
  
    # unc_test = pd.read_csv('testDataSet/unc.edu_mix8_IlluminaHiSeq_RNASeqV2.geneExp2.tsv',delimiter='\t', index_col=["gene_id"])
    # transpose the data to give samples in row and features in column
    unc_test_t = unc_test.T
    # replace ? with NA in gene ids
    unc_test_t.columns = unc_test_t.columns.str.replace('\?\|', 'NA|')
    n_sample = len(unc_test_t)
    st.write("Number of samples: ", n_sample)
    st.session_state['unc_test_t'] = unc_test_t
    st.session_state['uploaded_file'] = uploaded_file
    

ClassifierPickleFile = './CancerRNASeqXGBoostClassifier.pickle'
# xgboost.set_config(verbosity=0)
# st.write(xgboost.__version__)

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
# load the best XGBoost model:
if not (os.path.isfile(ClassifierPickleFile)):
    uploaded_model_file = st.file_uploader('Upload XGB Classifier Pickle File:', type=['pickle', 'pkl', 'p'], help='select the trained classfier model (.pickle)')
    if uploaded_model_file is not None: 
        ClassifierPickleFile = uploaded_model_file.name
    
if os.path.isfile(ClassifierPickleFile):
    best_xgb_model = pd.read_pickle(ClassifierPickleFile)
    st.session_state['best_xgb_model'] = best_xgb_model
    model_date = datetime.strptime(best_xgb_model['model_build_day'], '%Y%m%d').date()
    st.write('The model was trained on ', model_date, ' with ',   best_xgb_model['num_features'],  ' genes used.') 
    st.write('The model accuracy is ', best_xgb_model['accuracy'])
    st.write('The gene list used for the classification: (', best_xgb_model['num_features'], ' genes used)', best_xgb_model['selected_features'])

    selected_features = best_xgb_model['selected_features']
 
    xgb = best_xgb_model['xgb_model']

    gene_id_csv = convert_df(pd.DataFrame(selected_features))

    st.download_button(
        label="Download gene list as CSV",
        data=gene_id_csv,
        file_name='cancer_type_classifier_gene_list.csv',
        mime='text/csv',
    )
else: 
    st.write('No trained classifer file. Please upload the trained classifier pickle file.')