#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Thu Sep 19 23:27:10 2019

@author: adityabansal
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler,normalize

import matplotlib.pyplot as plt 
#Reading CSV file 
df=pd.read_csv("/Users/adityabansal/Desktop/ML/wdbc.csv", header=None )

#Replacing B&M to 0&1
df.replace(to_replace='B',
    value=0,inplace=True)
df.replace(to_replace='M',
    value=1,inplace=True)



#Slicing ID from the dataset and placing it in to redundant and other data in x
redundant= df.iloc[:,0]
x=df.iloc[:,1:]




#splitting our data in test,train,val

x_set ,x_train = train_test_split(x, train_size=0.20,random_state=25)
x_test ,x_validate = train_test_split(x_set, test_size=0.50,random_state=25 )


        



#slicing train data into train_Inst=instances and train_Sol=output(0 or 1)
train_Inst=x_train.iloc[:,1:]
train_Sol=x_train.iloc[:,0]



#slicing validated data into val_Inst=instances and val_Sol=output(0 or 1)
val_Inst=x_validate.iloc[:,1:]
val_Sol=x_validate.iloc[:,0]

#slicing validated data into test_Inst=instances and test_Sol=output(0 or 1)
test_Inst=x_validate.iloc[:,1:]
test_Sol=x_validate.iloc[:,0]


#Normalizing trian data(instances)

scaler = StandardScaler()
train_Inst = scaler.fit_transform(train_Inst)
train_Inst=train_Inst.T
print(train_Inst)


#Normalizing validate data(instances)

val_Inst = scaler.transform(val_Inst)
val_Inst=val_Inst.T


#Normalizing test data(instances)

test_Inst = scaler.transform(test_Inst)
test_Inst=test_Inst.T


#Reshaping our train_Sol array 
train_Sol = train_Sol.values
train_Sol = train_Sol.reshape(1, train_Sol.shape[0])


#Reshaping our val_Sol array 
val_Sol = val_Sol.values
val_Sol = val_Sol.reshape(1, val_Sol.shape[0])

#Reshaping our test_Sol array 
test_Sol = test_Sol.values
test_Sol= test_Sol.reshape(1, test_Sol.shape[0])





#Giving value of epochs which we have also chosen our hperparameter(1)
epochs =2000
#Giving value of learning rate which we have also chosen our hperparameter(2)
learningrate = 0.001

#Defining Sigmoid Function
def sigmoid(z):
      return 1 / (1 + np.exp(-z)) 
  
#Losstrack array which will store value of loss after every Iteration    
losstrack = []

#train_Accuracy array which will store value of training accuracy after every Iteration
train_Accuracy=[]

#val_Accuracy  array which will store value of validation accuracy after every Iteration
val_Accuracy=[]
losstrack_val=[]
# m is storing total no of instances we have taken in train set
m = train_Inst.shape[1]
m_val=val_Inst.shape[1]


#Instantiated weight array with 30 random values  
w = np.random.randn(train_Inst.shape[0], 1)*0.01

#Instantiated bias variable with zero 
bias = 0 

#Logistic regression training
for epoch in range(epochs):
        #Linear regression equation
        z_train = np.dot(w.T, train_Inst) + bias
        p_train = sigmoid(z_train)
        
        z_val = np.dot(w.T, val_Inst) + bias
        p_val = sigmoid(z_val)
        
        #Difference in predicted and given solution
        dz = p_train-train_Sol
        #Gradient Descent
        dw = (1 / m) * np.dot(train_Inst, dz.T)
        
        dbias = (1 / m) * np.sum(dz)
        # adjusting weight for every epoch
        w = w - learningrate * dw
        bias = bias - learningrate * dbias
        #Using cosr-entropy function finding cost for both train data and validation data
        cost = -np.sum(np.multiply(np.log(p_train), train_Sol) + np.multiply((1 - train_Sol), np.log(1-p_train)))/m
        losstrack.append(np.squeeze(cost))
        
        
        cost_val = -np.sum(np.multiply(np.log(p_val), val_Sol) + np.multiply((1 - val_Sol), np.log(1-p_val)))/m_val
        losstrack_val.append(np.squeeze(cost_val))
        #to find accuracy score we need predicted values in form of 0 and 1
        q=p_train
        q[q>=0.5]=1
        q[q<0.5]=0
        q=q.astype(int)
        #Finding accuracy score
        train_Accuracy.append(accuracy_score(y_true= train_Sol[0],y_pred= q[0]))
        
        p_val[p_val>=0.5]=1
        p_val[p_val<0.5]=0
        p_val=p_val.astype(int)
        val_Accuracy.append(accuracy_score(y_true= val_Sol[0],y_pred= p_val[0]))
#Plotting graph of cost vs epochs       
plt.plot(losstrack,label="train loss")
plt.legend(loc="upper right")


plt.plot(losstrack_val, label="validation loss")
plt.legend(loc="upper right")

plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
#Plotting graph for accuracy score vs epochs
plt.plot(train_Accuracy,label="training accuracy")
plt.legend(loc="lower right")
plt.plot(val_Accuracy,label="validation accuracy")
plt.legend(loc="lower right")
plt.xlabel("epochs")
plt.ylabel("training accuracy")
plt.show()


#Finding predictes values using sigmoid for test data
z_test = np.dot(w.T, test_Inst) + bias
p_test = sigmoid(z_test)

p_test[p_test>=0.5]=1
p_test[p_test<0.5]=0
p_test=p_test.astype(int)
#Finding accuracy score for test data
print(accuracy_score(y_true= test_Sol[0],y_pred= p_test[0]))

#Finding precision,recall,f1-score for train,validation and test data sets
print("Train Data Results:-")
print(classification_report(train_Sol[0], q[0]))
print("Validation Data Results:-")
print(classification_report(val_Sol[0],p_val[0]))
print("Test Data Results:-")
print(classification_report(test_Sol[0],p_test[0]))

