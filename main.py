# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:19:55 2020

@author: sameer
#coronavirus predictor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import keras

#reading the dataset

dataset_train = pd.read_csv('corona.csv')
dataset_train.head()
training_set = dataset_train.iloc[:,0:7].values
#scaling the dataset
spain_cases = []
spain_deaths = []
spain_dr = []
spain_ir = []
for i in range(len(training_set)):
    if training_set[i][1]=='30-39' and training_set[i][2]=='ambos' :
        total = training_set[i][3] + training_set[i+1][3] 
        total_deaths = training_set[i][6] + training_set[i+1][6]
        death_rate = total_deaths/total * 100
        infection_rate = total/20000000 * 100
        temp = []
        temp2 = []
        temp3= []
        temp4 = []
        #temp.append(training_set[i][0])
        temp.append(total)
        temp2.append(total_deaths)
        temp3.append(death_rate)
        temp4.append(infection_rate)
        spain_cases.append(temp)
        spain_deaths.append(temp2)
        spain_dr.append(temp3)
        spain_ir.append(temp4)

Spain_cases = np.array(spain_cases)
Spain_deaths = np.array(spain_deaths)
Spain_dr = np.array(spain_dr)
Spain_ir = np.array(spain_ir)
#print(spain)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(Spain_ir)

#preprocessing

X_train = []
y_train = []
for i in range(20, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#building lstm neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
regressor = Sequential()

regressor.add(LSTM(units = 25 , return_sequences = True, input_shape = (X_train.shape[1],1) ))
regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 25 , return_sequences = True ))
# regressor.add(Dropout(0.2))
regressor.add(Flatten())
# regressor.add(LSTM(units = 25 , return_sequences = True ))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 25 ))
# regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error',metrics = ['accuracy'])

regressor.fit(X_train, y_train, epochs = 100, batch_size =32)

#storing the predicted values

X_last = X_train[-1].tolist()
temp = X_last.copy()
final_cases = {}
for i in range(65):
    X_last2 = [ ]
    X_last2.append(temp)
    X_last2 = np.array(X_last2)
    X_last2 = np.reshape(X_last2, (X_last2.shape[0],X_last2.shape[1],1))
    predicted_cases = regressor.predict(X_last2)
    cases_i = sc.inverse_transform(predicted_cases)
    final_cases[i] = float(cases_i[0][0])
    temp = []
    temp = X_last[1:20]
    print(predicted_cases)
    temp.append(predicted_cases[0])
    X_last = temp.copy()
    

final_values = {}
final_values = (final_cases.values())
print(final_values)
final_values = np.array(list(final_values)).astype(float)
third = []
third = final_values[19:65]

print(third)

#infection = []
#for i in range(third):
    #ans=[]
    #ans.append(third[i]/10000) 
    #infection.append(ans)

# formatter = mdates.DateFormatter('%m/%d')
# date1 = mdates.datetime.date(2020, 2, 15)
# date2 = mdates.datetime.date(2020, 5, 21)
# delta = mdates.datetime.timedelta(days=65)
# dates = mdates.drange(date1, date2, delta)
#Visualising the results

plt.plot(third,color = 'red' , label = 'Infectionrate in age group 35-50 till 31st JULY')
# plt.plot_date(dates,final_values)
plt.title('Infectionrate in age group 35-50 till 31st JULY')
plt.xlabel('Days')
plt.ylabel('% percent')
plt.legend()
plt.show()    