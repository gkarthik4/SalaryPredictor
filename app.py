#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template 
import pickle


# In[23]:


app = Flask(__name__)


# In[24]:


filename = 'price_predictor_model.sav'
model = pickle.load(open(filename,'rb'))


# In[25]:


'''
    dayhours = 20141107
    room_bed = 4
    room_bath = 3
    living_measure = 3020
    lot_measure = 13457
    ceil = 1
    coast = 0
    sight = 0
    condition = 5
    quality = 9
    ceil_measure = 3020
    basement = 0
    yr_built = 1956
    yr_renovated = 0
    zipcode = 98133
    lat = 47
    long = 122
    living_measure15 = 2120
    lot_measure15 = 7553
    furnished = 1
    total_area = 16477
    
    data = [dayhours,room_bed,room_bath,living_measure,lot_measure,ceil,coast,sight,condition,quality,ceil_measure,basement,yr_built,yr_renovated,zipcode,lat,long,living_measure15,lot_measure15,furnished,total_area] 
    df = pd.DataFrame([data])
    df.columns = ['dayhours','room_bed','room_bath','living_measure','lot_measure','ceil','coast','sight','condition','quality','ceil_measure','basement','yr_built','yr_renovated','zipcode','lat','long','living_measure15','lot_measure15','furnished','total_area']
    
    df['Date'] = df['dayhours'].astype('datetime64[ns]')
    df['sold_in_yr'] = pd.DatetimeIndex(df['Date']).year
    df =df.drop(labels= "Date" , axis = 1)
    df =df.drop(labels= "dayhours" , axis = 1)
    df['house_age'] = df['sold_in_yr'] - df['yr_built'].astype(int)
    df['is_renovated'] = np.where(df.yr_renovated == 0,0,1)
    df =df.drop(labels= "yr_renovated" , axis = 1)
'''


# In[26]:


#df.head()


# In[27]:


#prediction = model.predict(df)
#print(prediction)
#print(np.expm1(prediction))


# In[28]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    dayhours = request.form.get('dayhours')
    room_bed = request.form.get('room_bed')
    room_bath = request.form.get('room_bath')
    living_measure = request.form.get('living_measure')
    lot_measure = request.form.get('lot_measure')
    ceil = request.form.get('ceil')
    coast = request.form.get('coast')
    sight = request.form.get('sight')
    condition = request.form.get('condition')
    quality = request.form.get('quality')
    ceil_measure = request.form.get('ceil_measure')
    basement = request.form.get('basement')
    yr_built = request.form.get('yr_built')
    yr_renovated = request.form.get('yr_renovated')
    zipcode = request.form.get('zipcode')
    lat = request.form.get('lat')
    long = request.form.get('long')
    living_measure15 = request.form.get('living_measure15')
    lot_measure15 = request.form.get('lot_measure15')
    furnished = request.form.get('furnished')
    total_area = request.form.get('total_area')
    
    
    #data = [dayhours,room_bed,room_bath,living_measure,lot_measure,ceil,coast,sight,condition,quality,ceil_measure,basement,yr_built,yr_renovated,zipcode,lat,long,living_measure15,lot_measure15,furnished,total_area] 
    data =  [float(x) for x in request.form.values()]
    df = pd.DataFrame([data])
    df.columns = ['dayhours','room_bed','room_bath','living_measure','lot_measure','ceil','coast','sight','condition','quality','ceil_measure','basement','yr_built','yr_renovated','zipcode','lat','long','living_measure15','lot_measure15','furnished','total_area']
    
    df['Date'] = df['dayhours'].astype('datetime64[ns]')
    df['sold_in_yr'] = pd.DatetimeIndex(df['Date']).year
    df =df.drop(labels= "Date" , axis = 1)
    df =df.drop(labels= "dayhours" , axis = 1)
    df['house_age'] = df['sold_in_yr'] - df['yr_built'].astype(int)
    df['is_renovated'] = np.where(df.yr_renovated == 0,0,1)
    df =df.drop(labels= "yr_renovated" , axis = 1)
    
    prediction = np.expm1(model.predict(df))
    #prediction = model.predict(df)

    output = round(prediction[0], 2)
    #output = prediction
    
    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))  


# In[ ]:


if __name__ == "__main__":
   
    app.run(debug = True)
    

# In[ ]:




