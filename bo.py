# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:31:23 2023

@author: David 
"""

import pandas as pd 
#import numpy as np
import pickle


df=pd.read_csv('DataFolder/Gold/hour.csv',header=0,index_col=0)

df.head()

df.columns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from bayes_opt import BayesianOptimization
import numpy as np

'''Data in Feature Matrix X and target variable y'''

X=df.drop(columns='cnt',axis=1)
y=df['cnt']

'''For computation on a notebook count the number of processors'''
import os
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

#Define Feature name
feature_names=[col for col in X.columns]

'''We split Data, such that we have a test set which the model has never
seen before. Our Train set will be used for training with cross validation'''

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)

'''Prediction with RandomForest regressor. 
 Tuning with Bayesian Optimization.'''

'''Function which defines the model. ''' 
def regressor(features,target,n_estimators,max_features,max_samples,
           crit='absolute_error',scoring='neg_mean_absolute_error',
           n_jobs=n_cpu):
    
    params = {'n_estimators': int(n_estimators),
              'max_features':max_features,
              'max_samples':max_samples
              }
    reg=RandomForestRegressor(criterion=crit,n_jobs=n_jobs,
                              max_samples=params['max_samples'],
                              max_features=params['max_features'],
                              n_estimators=params['n_estimators'])
    score=cross_val_score(reg,features,target,
                          scoring=scoring,
                          n_jobs=n_jobs,cv=5).mean()
    return score

'''Define the BO objective functions.'''
def bo_obj(n_estimators,max_features,max_samples):
    return regressor(X_train, y_train, n_estimators, max_features,
                     max_samples,crit="squared_error")

'''Define the Optimizer'''
rf_bo = BayesianOptimization(bo_obj,{
                                              'max_samples':(0.75,1),
                                                'max_features':(0.75,1),
                                              'n_estimators':(100,501),
                                              }, random_state=42)
'''Due to hardware constraint, I run only 5 iterations
and chose the squared_error split criterion'''
results = rf_bo.maximize(n_iter=8, init_points=5)

print(rf_bo.max)

'''Define and Fit Model with optimal 
parameter derived from Bayesian Opt '''

params_opt = rf_bo.max['params']

rf_opt= RandomForestRegressor(n_estimators=int(params_opt['n_estimators']),
                              criterion="squared_error",n_jobs=n_cpu-1,
                              max_samples=params_opt['max_samples'],
                              max_features=params_opt['max_features'])

'''Compute the mean absolute deviations of the model with cross valdiation'''

scores=cross_val_score(rf_opt,X_train,y_train,scoring="neg_mean_absolute_error",cv=5,n_jobs=n_cpu-1)
print(np.round(-scores,2))
print(f''' The estimated generalized mean absolute error of the final model on
      via cross validation is {np.round(-scores.mean(),2)}''')

'''Fit the Model to save it '''
rf_opt.fit(X_train,y_train)

'''Save the model for later prediction purposes'''
filename = 'Model/model_tuned.sav'
pickle.dump(rf_opt, open(filename, 'wb'))


'''Load the Model and compute MAE on unseen Data'''
model=pickle.load(open(filename, 'rb'))
predictions = model.predict(X_test)
test_result = mae(predictions, y_test)

print(f''' The mean absolute error of the final model on
      the unseen data is {np.round(test_result,2)}''')
      
'''Prepare the Predictions on the unseen Data'''

pred=pd.DataFrame(y_test)
pred['y_hat']=predictions
pred.reset_index(inplace=True,drop=True)

#Plot the Result

ax = pred.plot(kind='scatter',x='cnt',y='y_hat')
ax.set_xlabel("True Demand")
ax.set_ylabel("Predicted Demand")
ax.set(title='Predicted vs Actual Bike Rentals')
fig=ax.get_figure()
fig.savefig("Plots/prediction.jpeg")


'''We run this Model for Business Case so we should
interpret the Results, use the SHAP Explainer for that'''

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train,approximate=True)
plt_shap = shap.summary_plot(shap_values, #Use Shap values array
                             features=X_train, # Use training set features
                             feature_names=X_train.columns, #Use column names
                             plot_size=(30,20))

                             
