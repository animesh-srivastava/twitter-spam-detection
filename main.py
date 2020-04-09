# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:50:44 2020

@author: animesh-srivastava
"""

#%% Importing the modules
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
#%% Importing and cleaning the data
df = pd.read_csv("train.csv")
df.drop(["Id","Tweet","location","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11"],axis=1,inplace=True)
df.dropna(inplace=True)
df = df[df["following"].apply(lambda x: x.isnumeric())]
df = df[df["followers"].apply(lambda x: x.isnumeric())]
df = df[df["actions"].apply(lambda x: x.isnumeric())]
df = df[df["is_retweet"].apply(lambda x: x.isnumeric())]
df = df[df["Type"].apply(lambda x: x=="Quality" or x=="Spam")]
#%% Label Encoder on the final column
enc = LabelEncoder()
df["Type"] = enc.fit_transform(df["Type"])
#%% Defining the input and the target values
x = df.iloc[:,0:4].values
y = df.iloc[:,4:5].values
sc = StandardScaler()
x = sc.fit_transform(x)

#%% Defining the model
def evaluate_model(x_train,x_test,y_train,y_test,count):
    model = Sequential()
    model.add(Dense(16,activation='relu',input_shape=[4]))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(x_train,y_train,batch_size = 5, epochs = 10,validation_data=(x_test,y_test))
    if not os.path.isdir("./weights/"):
        os.mkdir("./weights/")
    model.save("./weights/model_part_"+str(count)+".hdf5")
    val, val_acc = model.evaluate(x_test, y_test)
    return model, val_acc
#%% K-fold cross validation
acc,model_list = list(),list()
n_folds = 10
count = 1
kfold = KFold(n_folds,shuffle=True)
for train,test in kfold.split(x,y):
    print("Running k fold cross validation training with k = "+str(n_folds)+". The Current count is "+str(count))
    model, val_acc = evaluate_model(x[train],x[test],y[train],y[test],count)
    count=count+1
    acc.append(val_acc)
    model_list.append(model)
#%% Measuring the accuracy
accuracy = np.mean(acc)
print(f'Accuracy is {np.mean(acc)*100}% ({np.std(acc)*100})')
#%% 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1./n_folds)
y_pred = model.predict(x_test)
y_pred = (y_pred>0.5)

#%%
cm = confusion_matrix(y_test, y_pred)
print(cm)
#%%
tdf = pd.read_csv('test.csv')
x_val = tdf.iloc[:,2:6].values
x_val = sc.transform(x_val)
y_val = model.predict(x_val)
tdf["Type"] = y_val>0.5
tdf["Type"].replace({True: 1, False: 0},inplace=True)
tdf["Confidence"] = y_val*100
tdf["Type"] = enc.inverse_transform(tdf["Type"])
tdf.to_csv("predicted.csv")
