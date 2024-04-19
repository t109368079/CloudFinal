# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:00:19 2024

@author: garyy
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder as le
from sklearn.ensemble import RandomForestRegressor as rfr




def swap(list_, p1, p2):
    list_[p1], list_[p2] = list_[p2], list_[p1]
    return list_

def select_col_name(df, name):
    col_names = list(df.columns)
    for col in col_names:
        if name in col:
            return col

#%% Q2 Login section
login_holder = st.empty()
if 'login' not in st.session_state.keys():
    with login_holder.form('Login'):
        st.markdown("#### Enter username, password and email")
        
        user_name = st.text_input("Enter user name")
        password = st.text_input('Enter your password')
        email = st.text_input('Enter your email')
        st.session_state['user_name'] = str(user_name)
        st.session_state['password'] = str(password)
        st.session_state['email'] = str(email)
        submit = st.form_submit_button("Login")
    
    if submit:
        user_df = pd.read_csv('./Data/password.csv')
        
        if st.session_state['user_name'] not in list(user_df['user_name']):
            temp = pd.DataFrame({'user_name':[st.session_state['user_name']], 'password':[st.session_state['password']], 'email':[st.session_state['email']]})
            user_df = pd.concat([user_df, temp])
            user_df.to_csv('./Data/password.csv', index=False)
            st.session_state['isNew'] = True
            login_holder.empty()
            st.session_state['login'] = True
        
        else:
            st.session_state['isNew'] = False
            user_data = user_df[user_df['user_name'] == st.session_state['user_name']]
            password = [str(i) for i in list(user_data['password'])]
            email = list(user_data['email'])
            if st.session_state['password'] not in password:
                st.error('WRONG PASSWORD! ARE YOU TRY TO HACK THE SYSTEM?')
            elif st.session_state['email'] not in email:
                st.error('WRONG EMAIL! ARE YOU TRY TO HACK THE SYSTEM?')
            else:
                login_holder.empty()
                st.session_state['login'] = True
            
    


#%% Page config
if 'login' in st.session_state.keys():
    if st.session_state['isNew']:
        st.warning(f"{st.session_state['user_name']}: Username create, email: {st.session_state['email']}")
        
    st.title("Cloud Computing Final project")
    
    #%% Ans to Q1
    text_path = './Q1.txt'
    with open(text_path, 'r') as f:
        Q1_str = f.read()
        
    st.header('Q1 Short write on ML model')
    st.text_area("Ans to Q1(Don't write anything!')", Q1_str)
    
    
    #%% Load data
    transaction_df = pd.read_csv('./Data/transaction.csv')
    product_df = pd.read_csv('./Data/product.csv')
    household_df = pd.read_csv('./Data/household.csv')
    
    #%% Q3 Sample data for HSHD_NUM == #10
    
    new_df = pd.merge(transaction_df, product_df)
    new_df = pd.merge(new_df, household_df)
    
    col_names = list(new_df.columns)
    col_names = swap(col_names, 9, 4)
    col_names = swap(col_names, 10, 5)
    new_df = new_df[col_names]
    
    select_name = 'HSHD_NUM'
    name = select_col_name(new_df, select_name)
    select_df = new_df[new_df[name] == 10]
    select_df = select_df.set_index(name)
    
    st.header("Q3 Table for HSHD_NUM #10")
    st.dataframe(select_df)
    #%% Q4 select HSHD num
    
    st.header("Q4 search based on HSHD num")
    hshd_list = list(household_df[select_col_name(household_df, 'HSHD_NUM')])
    select_hshd = st.selectbox("Select HSHD here", hshd_list)
    
    select_name = 'HSHD_NUM'
    name = select_col_name(new_df, select_name)
    select_df = new_df[new_df[name] == select_hshd]
    select_df = select_df.set_index(name)
    st.dataframe(select_df)
    
    
    #%% Q5 Which factor affect custom engagement
    
    st.header("Q5 Show which factor affecrt custom engagement")
    # Step 1. Create total spend for all household
    name = select_col_name(household_df, 'CHILDREN')  
    household_df[name] = household_df[name].fillna('0')
    household_clean_df = household_df.dropna()
    
    hshd_list = list(household_clean_df[select_col_name(household_clean_df, 'HSHD_NUM')])
    
    spend_dict = []
    for hshd in hshd_list:
        name = select_col_name(transaction_df, 'HSHD_NUM')
        hs_df = transaction_df[transaction_df[name] == hshd]
    
        total_spend = hs_df[select_col_name(hs_df, 'SPEND')].sum()
        spend_dict.append(total_spend)
        
    
    
    spend_df = pd.DataFrame({name: hshd_list, 'total_spend':spend_dict, 'count':1})
    household_clean_df = pd.merge(household_clean_df, spend_df)
    
    # Step 2. Remove null data
    for col_name in list(household_clean_df.columns):
        remove_index = household_clean_df[household_clean_df[col_name] == 'null   '].index
        household_clean_df = household_clean_df.drop(remove_index)
        
    # Step 3. label encoder
    encoder = le()
    encode_col = list(household_clean_df.columns)
    encode_col.remove('total_spend')
    
    household_encoded_df = household_clean_df.copy()
    for col in encode_col:
        household_encoded_df[col] = encoder.fit_transform(household_clean_df[col])
        
    # Step 4. Plot heat map
    corr = household_encoded_df.corr()
    corr_fig = px.imshow(corr, text_auto=True)
    corr_fig.update_layout(title='Correlation Matrix')
    st.plotly_chart(corr_fig)
    
    # Step 4.1 用文字說明
    st.text_area("Ans to Q5",
                'Based on the correlation heatmap showing above, total spend has highly related with 1. Loyalty and 2. Children and household size'+
                'Loyalty and household size are affect to customer engagement'
                )
    st.write('Figure below shows loyalty vs total spend and household size vs total spend')
    # Step 5. Bar plot in plotly for 1) Lotaly vs total spend, 2) household size vs total spend and 3) children vs total spend
    col1, col2 = st.columns(2)
    
    
    ly_sum = household_clean_df.groupby([select_col_name(household_clean_df, 'L')]).sum()
    ly_fig = go.Figure()
    trace = go.Bar(
        x = ['N', 'Y'],
        y = ly_sum['total_spend']/ly_sum['count'])
    ly_fig.add_trace(trace)
    ly_fig.update_layout(title='Loalty vs total spend')
    st.plotly_chart(ly_fig)
    
    
    hsize_sum = household_clean_df.groupby([select_col_name(household_clean_df, 'HH_SIZE')]).sum()
    hsize_fig = go.Figure(go.Bar(
        x=['1','2','3','4','5+'],
        y = hsize_sum['total_spend']/hsize_sum['count']))
    hsize_fig.update_layout(title='household size vs total spend')
    st.plotly_chart(hsize_fig)
    
    #%% Q6 Upload new file
    
    st.header("Q6 Upload new file")
    
    # Step 6.1 Init session_state
    if 'household_uploaded' not in st.session_state.keys():
        st.session_state['household_uploaded'] = None
    if 'transaction_uploaded' not in st.session_state.keys():
        st.session_state['transaction_uploaded'] = None
    if 'product_uploaded' not in st.session_state.keys():
        st.session_state['product_uploaded'] = None
    
    
    # Step 6.2 File uploader
    household_uploader = st.file_uploader("Upload household file")
    transaction_uploader = st.file_uploader("UPload transaction file")
    product_uploader = st.file_uploader('Upload product file')
    
    if household_uploader is not None:
        new_house_df = pd.read_csv(household_uploader)
        st.session_state['household_uploaded'] = new_house_df
        
    is_house = True
    if st.session_state['household_uploaded'] is None:
        st.warning("NEW HOUSEHOLD FILE NOT UPLOADED !!!")
        is_house = False
    
    if transaction_uploader is not None:
        new_trans_df = pd.read_csv(transaction_uploader)
        st.session_state['transaction_uploaded'] = new_trans_df
        
    is_trans = True
    if st.session_state['transaction_uploaded'] is None:
        st.warning("NEW TRANSACTION FILE NOT UPLOADED !!!")
        is_trans = False
    
    if product_uploader is not None:
        new_product_df = pd.read_csv(product_uploader)
        st.session_state['product_uploaded'] = new_product_df
    
    is_prod = True
    if st.session_state['product_uploaded'] is None:
        st.warning("NEW PRODUCT FILE NOT UPLOADED !!!")
        is_prod = False
    
    # Step 6.4 Process if all files are uploaded
    if is_house and is_trans and is_prod:
        new_house_df = st.session_state['household_uploaded']
        new_trans_df = st.session_state['transaction_uploaded']
        new_product_df = st.session_state['product_uploaded']
        uploaded_df = pd.merge(new_trans_df, new_product_df)
        uploaded_df = pd.merge(uploaded_df, new_house_df)
        
        hshd_list = list(new_house_df[select_col_name(new_house_df, 'HSHD_NUM')])
        select_hshd_num = st.selectbox("Select HSHD here", hshd_list, key='selectbox2')
    
        select_name = 'HSHD_NUM'
        name = select_col_name(uploaded_df, select_name)
        select_df = uploaded_df[uploaded_df[name] == select_hshd_num]
        select_df = select_df.set_index(name)
        st.dataframe(select_df)
        
#%% Q7 Machine learning method 
    st.header("Extra, ML method for factor affect customer engagement")
    regr = rfr(max_depth=3, random_state=72)
    feature_col = list(household_encoded_df.columns)
    feature_col = feature_col[:-2]
    feature = household_encoded_df[feature_col]
    label = household_encoded_df['total_spend']
    
    # Change columns name 因為有智障不知道怎麼搞得把col name打了一堆空格，幹
    ori_col = list(feature.columns)
    new_col = [each.replace(' ', '') for each in ori_col]
    feature.columns = new_col
    
    regr.fit(feature, label)
    
    # Plot tree
    estimator = regr.estimators_[0]
    fig = plt.figure(figsize=(10,6))
    plot_tree(estimator, feature_names=feature.columns, class_names=label.name, filled=True)
    
    st.pyplot(fig)
    
    # Explaination
    st.text_area("Ans for Extra", "According to the random forest analysis, the tree initially divides based on the attribute L"+ 
        "representing loyalty program enrollment, and subsequently by attribute HH_SIZE, indicating household size."+
        "As indicated by the heatmap in Q5, total spending demonstrates a strong correlation with these two attributes, as confirmed by the random forest results.\n"+
        "In summary, the two most significant factors influencing customer engagement are Loyalty (L) and Household Size (HH_SIZE).")
