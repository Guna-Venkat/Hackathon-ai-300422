import streamlit as st
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans,DBSCAN,MiniBatchKMeans,MeanShift,SpectralClustering,OPTICS,AgglomerativeClustering
from sklearn.decomposition import PCA,NMF,KernelPCA,MiniBatchSparsePCA
from sklearn.feature_selection import RFE
import pickle
import math
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = pickle.load(open('Final.sav','rb'))
def predict(X):
    pred = model.predict(X)
    return pred
def main():
    crim = st.number_input('CRIM',0.00,100.00)
    zn = st.number_input('ZN',0.00,100.00)
    indus = st.number_input('INDUS',0.00,30.00)
    chas = st.number_input('CHAS',0.00,1.00)
    nox = st.number_input('NOX',0.00,1.00)
    rm = st.number_input('RM',2.00,10.00)
    age = st.number_input('AGE',0,100)
    dis = st.number_input('DIS',0.00,15.0)
    rad = st.number_input('RAD',1.00,25.00)
    tax = st.number_input('TAX',150.00,750.00)
    ptr = st.number_input('PTRATIO',10.00,25.00)
    b = st.number_input('B',0.00,400.00)
    lstat = st.number_input('LSTAT',1.00,40.00)
    df = pd.DataFrame({'CRIM':crim,'ZN':zn,'INDUS':indus,'CHAS':chas,'NOX':nox,
                       'RM':rm,'AGE':age,'DIS':dis,'RAD':rad,'TAX':tax,'PTRATIO':ptr,
                      'B':b,'LSTAT':lstat},columns=['CRIM' ,'ZN' ,'INDUS' ,'CHAS' ,'NOX' ,'RM' ,'AGE' ,'DIS' ,'RAD' ,'TAX' ,'PTRATIO','B' ,'LSTAT'],index=[0])
    X = df
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
    df['new_1'] = kmeans.labels_
    db = DBSCAN().fit(X)
    df['new_2'] = db.labels_
    mb = MiniBatchKMeans(n_clusters=1).fit(X)
    df['new_3'] = mb.labels_
    ms = MeanShift().fit(X)
    df['new_4'] = ms.labels_
    #op = 0
    df['new_5'] = 0
    X_pca = PCA().fit_transform(X)
    df['new_6'] = X_pca
    X_kpca = KernelPCA().fit_transform(X)
    df['new_7'] = sum(X_kpca[0])
    X_mbpca = MiniBatchSparsePCA().fit_transform(X)
    df['new_8'] = sum(X_mbpca[0])
    X = df[['CRIM', 'INDUS', 'NOX', 'RM', 'DIS', 'TAX', 'PTRATIO', 'LSTAT', 'new_6','new_8']]
    
    if st.button('PREDICT'):
        out = predict(X)
        st.success(out)

if __name__=='__main__':
    main()