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
import plotly.express as px

import pandas as pd
import numpy as np

import xgboost
from xgboost import XGBClassifier

import pickle
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("Predicted Cancer Type Distribution")

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
    if uploaded_file is not None:
        st.write("This result is for the data set - " + uploaded_file.name)
if 'result_df' in st.session_state:        
    result_df = st.session_state['result_df']  
  
    # Cancer class distribution among the UNC data set
    fig = px.histogram(result_df, x = 'Predicted Cancer Type')
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)


    # result_df = result_df_org.drop(columns='Cancer Type Description')
    st.write(result_df) 
else: 
    st.write("Warning - no prediction result. Please go to last two steps")