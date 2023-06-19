# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:12:08 2023

@author: David
"""

# Import required Libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Read in the dataset, having in a folder called
 DataFolder/Bronze  where DataFolder is in the 
 same directory as the script'''

df = pd.read_csv("DataFolder/Bronze/hour.csv",header=0)

#Inspect DataFrame
print(df.head(5))
print(df.info())
#Check datatypes
print(df.dtypes)


'''We should convert columns like season, workingday, holiday ...
into categorical, which we do later. dteday provides no new information, as we have
mnth, day ,yr ,hour but we keep it and convert it to datetime'''

df['datetime']=pd.to_datetime(df['dteday'])
df.drop(columns=['dteday'],axis=1,inplace=True)

#Look at summary stats
print(df.describe())

df['cnt'].describe()


'''Save the summary stats'''

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

pd.plotting.table(ax, np.round(df['cnt'].describe(),3))  # where df is your data frame

plt.savefig('Plots/mytable.jpeg')

'''Visualize the Bike Rentals on date axis'''
ax=df.plot(x='datetime',y='cnt',figsize=(16,8),title='Bike Rental over Time')
fig=ax.get_figure()
fig.savefig("Plots/date_cnt.jpeg")

'''We look at the histogramm and Boxplot of the dependent Variable'''


fig,axes =plt.subplots(1,2)
plot=sns.histplot(ax=axes[0],data=df[['cnt']],bins=15,label='hist',kde=True)
axes[0].set(xlabel='Bike Demand', ylabel='Frequency')
axes[0].legend().remove()
sns.boxplot(ax=axes[1],data=df[['cnt']])
fig.savefig("Plots/cnt_hist.jpeg")


'''Add the #Bike Rentals from one hour ago as feature'''
df['lag_cnt'] = df['cnt'].shift(1)


'''Drop the first row and convert columns into int'''
df.drop(0,axis=0,inplace=True)
df['lag_cnt'].astype('int')


'''Hour is  cyclic data -> transform it. '''

df['hr_sin'] = np.sin(2 * np.pi * df['hr']/24.0)
df['hr_cos'] = np.cos(2 * np.pi * df['hr']/24.0)

'''Visualize the Effect'''
ax=df.sample(80).plot(title='Cyclic Transformation of Hour',kind='scatter',x='hr_sin',y='hr_cos')
ax.set_aspect('equal')
fig=ax.get_figure()
fig.savefig("Plots/hr_cyclic.jpeg")


'''Now we typecast some columns in caterogical'''

categorical=['season','yr', 'mnth','holiday', \
             'weekday', 'hr','weathersit','workingday']   

for col in categorical:
    df[col]= df[col].astype('category')
    





print(f"Dimension of Dataset is {df.shape}")

'''We have 17379 observations, for two years and hourly data
should been around 17520 -> Check if missing is balanced 
and visualize the impacts of several features on Rentals '''

'''Functions that saves the plots to path_to_save 
in the specified format'''
def plotting(path_to_save,format='jpeg'):
    
    fig, axes = plt.subplots(3, 2, figsize=(22, 14))

    fig.suptitle('Bike Demand',fontsize=14)

    df[['season','cnt']].groupby('season')['cnt'].sum().plot.bar(ax=axes[0, 0], x='season',y='cnt',title='Bike Rentals per Season')
    axes[0,0].set(xticklabels=['Winter','Spring','Summer','Fall'])
    df[['yr','cnt']].groupby('yr')['cnt'].sum().plot.bar(ax=axes[0, 1], x='yr',y='cnt',title='Bike Rentals per Year')
    df[['mnth','cnt']].groupby('mnth')['cnt'].sum().plot.bar(ax=axes[1, 0], x='mnth',y='cnt',title='Bike Rentals per Month')
    df[['weathersit','cnt']].groupby('weathersit')['cnt'].sum().plot.bar(ax=axes[1, 1], x='weathersit',y='cnt',title='Bike Rentals on Weather')
    axes[1,1].set(xticklabels=['Clear','Mist','Light Rain/Snow','Heavy Rain/Snow'])
    df[['weekday','cnt']].groupby('weekday')['cnt'].sum().plot.bar(ax=axes[2, 0], x='weekday',y='cnt',title='Bike Rentals per Weekday')
    axes[2,0].set(xticklabels=['Su','Mo','Tu','Wed','Thu','Fri','Sa'])
    df[['hr','cnt']].groupby('hr')['cnt'].sum().plot.bar(ax=axes[2, 1], x='hr',y='cnt',title='Bike Rentals per Hour')
    fig.savefig(path_to_save + f".{format}")

plotting("Plots/cnt_per_features")

'''The Plots for hours indicate impact of working times on bike rental demand.
We analyze it with plotting against workday'''

def plot_work(path_to_save,format='jpeg'):
    fig,ax=plt.subplots(figsize=(15,8))
    #Groupby workingday and season for the sums
    tmp=pd.DataFrame(df[['workingday','season','cnt']].groupby(['workingday','season'])['cnt'].sum())
    #unstack the indices
    tmp.unstack().plot(ax=ax,kind='bar',xlabel='Workingday',ylabel='Bike Rentals',title='Bike Rental Demand per Season dep. on Workingday')
    ax.legend(['Winter','Spring','Summer','Fall'],loc='upper left',fontsize=15)
    ax.set(xticklabels=['No','Yes'])
    fig.savefig(path_to_save + f".{format}")
    
plot_work("Plots/Workingday")

'''Visualize Impact of temp, humidity,windspeed on cnt'''

def plot_condit(path_to_save,format='jpeg'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    fig.suptitle('Bike Demand on Temperature Conditions',fontsize=15)

    df.plot(ax=axes[0],kind='scatter', x='hum',y='cnt',title='Bike Rentals against Humidity')
    df.plot(ax=axes[1],kind='scatter', x='temp',y='cnt',title='Bike Rentals against Feeled Temperature')
    df.plot(ax=axes[2],kind='scatter', x='windspeed',y='cnt',title='Bike Rentals against Windspeed')
    fig.savefig(path_to_save + f".{format}")
    
plot_condit("Plots/weather_cond")

'''Further Clean the Dataset'''

#Col instant is just an id -> drop it 
df.drop(columns="instant",axis=1,inplace=True)


'''First Glance indicated that
 casual + registered = cnt
 We check their correlation'''

print(f'The correlation between registered + casual and cnt is {np.corrcoef(df["casual"]+df["registered"],df["cnt"])[0,1]}')


#We can hence drop casual and registered

df.drop(columns=["casual","registered"],axis=1, inplace=True)

'''Check for other features with high correlation'''

plt.figure(figsize=(14, 6))
corr = sns.heatmap(df.corr(), cmap="flare", annot=True)
corr.set_title('Correlation', fontsize=15, pad=12);
fig=corr.get_figure()
fig.savefig("Plots/corr.jpeg")

#Remove temp or atemp
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x="temp", y="atemp", data=df)
fig2=ax.get_figure()
fig2.savefig("Plots/temp.jpeg")

#We remove atemp as there are some incosistencies
df.drop(columns=["atemp"],axis=1, inplace=True)

#Check if there are columns with missing values

print(df.isnull().sum())

'''We have no missing values so we could keep this 
Dataframe for further  Visualizations 
or other Selections -> save it under a 
DataFolder/Silver'''

df.to_csv('DataFolder/Silver/hour.csv')


'''Now we clean the data for prediction purpose'''

#Check binary features
print(df[categorical].nunique())

'''Define function which takes a list of columns and 
deletes them'''

def col_to_drop(col_list):
    df.drop(col_list,axis=1,inplace=True)

'''Delete datetime,hr and month as the latter basically has same info
as season'''
delete=['datetime','hr','mnth','yr']
col_to_drop(delete)

'''Encode the categorical features. For the  ones with only 2 
categories we only keep one'''
cat=list(df.select_dtypes(['category']).columns)

binary= cat[1:3]+[cat[-2]]
cat=[x for x in cat if x not in binary]
df = pd.get_dummies(df,columns=cat,drop_first=False)
df = pd.get_dummies(df,columns=binary,drop_first=True)
df.info()

'''We save this in a Gold folder under the DataFolder'''

df.to_csv('DataFolder/Gold/hour.csv')


