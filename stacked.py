# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:58:31 2019

@author: dbda3
"""
###############################################################################
#stacking
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

abalone = pd.read_csv(r"D:\abalone_.data")
abalone.columns=["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]

abalone_dum = pd.get_dummies(abalone,drop_first=True)
y = abalone_dum["Rings"]
X=abalone_dum.drop(["Rings"],axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
y_train2=y_train
#### Model-1 linear regression  ####
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
pred_lr = model_lr.predict(X_train)
pred_lr [pred_lr <0] = 0
pred=model_lr.predict(X_test)
print(r2_score(y_test,pred))

#### Model-2 SVR 'linear'  ######
from sklearn.svm import SVR
model_svrl = SVR(kernel='linear')
model_svrl.fit(X_train,y_train)
pred_svrl=model_svrl.predict(X_train)
pred_svrl[pred_svrl <0] = 0
print(r2_score(y_train,pred_svrl))
pred = model_svrl.predict(X_test)
print(r2_score(y_test,pred))
#### Model-3 SVR 'radial' ######
model_svrr = SVR(kernel='rbf')
model_svrr.fit(X_train,y_train)
pred_svrr=model_svrr.predict(X_train)
pred_svrr[pred_svrr <0] = 0
print(r2_score(y_train,pred_svrr))
pred=model_svrr.predict(X_test)
print(r2_score(y_test,pred))

#### Model-4 Decision Tree Regressor ######
from sklearn.tree import DecisionTreeRegressor
model_dtr= DecisionTreeRegressor()
model_dtr.fit(X_train,y_train)
pred_dtr=model_dtr.predict(X_train)
pred_dtr[pred_dtr<0]=0
print(r2_score(y_train,pred_dtr))
pred=model_dtr.predict(X_test)
print(r2_score(y_test,pred))

###### Combining all the predictions #####
pred_lr=pd.Series(pred_lr)
pred_svrl=pd.Series(pred_svrl)
pred_svrr=pd.Series(pred_svrr)
pred_dtr=pd.Series(pred_dtr)
comb_pred=pd.concat([pred_lr,pred_svrl,pred_svrr,pred_dtr],axis=1)
#(pred_lr,pred_svrl,pred_svrr,pred_svrs,pred_dt
comb_pred.columns=['pred_lr','pred_svrl','pred_svrr','pred_dtr']

###### Now level 2 model RF ############################################################
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

print(np.isnan(comb_pred))
print(np.isinf(comb_pred))
model.fit(comb_pred,y_train)


from sklearn.model_selection import train_test_split

abalone = pd.read_csv(r"D:\abalone_.data")
abalone.columns=["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]

abalone_dum = pd.get_dummies(abalone,drop_first=True)
y = abalone_dum["Rings"]
X=abalone_dum.drop(["Rings"],axis=1)


from sklearn.preprocessing import StandardScaler




scaler = StandardScaler()
scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=11)
pred_lr = model_lr.predict(X_test)
#pred_lr [pred_lr <0] = 0

pred_svrl=model_svrl.predict(X_test)
#pred_svrl[pred_svrl <0] = 0

pred_svrr=model_svrr.predict(X_test)
#pred_svrr[pred_svrr <0] = 0

pred_dtr=model_dtr.predict(X_test)
#pred_dtr[pred_dtr<0]=0


###### Combining all the predictions for test set #####
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

pred_lr=pd.Series(pred_lr)
pred_svrl=pd.Series(pred_svrl)
pred_svrr=pd.Series(pred_svrr)
pred_dtr=pd.Series(pred_dtr)
comb_pred_test=pd.concat([pred_lr,pred_svrl,pred_svrr,pred_dtr],axis=1)
#(pred_lr,pred_svrl,pred_svrr,pred_svrs,pred_dt
comb_pred_test.columns=['pred_lr','pred_svrl','pred_svrr','pred_dtr']

pred_testdata=model.predict(comb_pred_test)
print(r2_score(y_test,pred_testdata))
print(mean_absolute_error(y_test,pred_testdata))
print(mean_squared_error(y_test,pred_testdata))

#from xgboost import XGBRegressor
#
#xgbr = XGBRegressor()
#xgbr.fit(comb_pred,y_train2)
#xgb_pred = xgbr.predict(comb_pred_test)
#print(r2_score(y_test,xgb_pred))


#from sklearn.neighbors import KNeighborsRegressor
#kn = KNeighborsRegressor(n_neighbors=5)
#kn.fit(X_train,y_train)
#kn_pred = kn.predict(X_test)
#print(r2_score(y_test,kn_pred))
