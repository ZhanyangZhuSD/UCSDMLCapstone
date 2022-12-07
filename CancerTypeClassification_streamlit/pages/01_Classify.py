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
import time

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("Cancer Classifier by RNA-Seq Gene Expression - Classification Results")

if 'unc_test_t' in st.session_state: 
    unc_test_t = st.session_state['unc_test_t']
    if unc_test_t is None:
        st.text("Error: Go back to Setup Classifier page to upload test data file!")    
else: 
    st.text("Error: Go back to Setup Classifier page to upload test data file!") 

cancer_description = st.session_state['cancer_description']
     
if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
    if uploaded_file is not None:
        st.write("Processing..."+ uploaded_file.name)

if ('best_xgb_model' in st.session_state) and ('unc_test_t' in st.session_state):
    best_xgb_model = st.session_state['best_xgb_model']
    if best_xgb_model is not None:
        selected_features = best_xgb_model['selected_features']
        xgb = best_xgb_model['xgb_model']
        unc_test_t_new = unc_test_t[selected_features]
        unc_test_t_pred = xgb.predict(unc_test_t_new)
        sample_list = unc_test_t_new.index.values.tolist()
        n=len(sample_list)
        result_df = pd.DataFrame(columns=['Sample ID', 'Predicted Cancer Type', 'Cancer Type Description'])
        for i in range(n):
            tmp = {'Sample ID': sample_list[i], 
                'Predicted Cancer Type': unc_test_t_pred[i],
                'Cancer Type Description': cancer_description[unc_test_t_pred[i]]}
            # st.write(tmp)
            tmp_df = pd.DataFrame([tmp])
            result_df = pd.concat([result_df, tmp_df])        
            # st.write(sample_list[i] + "\t", unc_test_t_pred[i] + "\t", cancer_description[unc_test_t_pred[i]])
        # st.write(result_df)    
        st.session_state['result_df'] = result_df
        
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        pred_result_csv = convert_df(result_df)

        st.download_button(
            label="Download Results",
            data=pred_result_csv,
            file_name='Cancer_Type_Prediction_For_' + uploaded_file.name + '.csv',
            mime='text/csv',
        )        
        # CSS to inject contained in a string
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(result_df)
    
