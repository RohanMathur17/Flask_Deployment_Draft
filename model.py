import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('iris.csv')

import sklearn
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn import metrics
from sklearn import svm 

df= df.drop(columns = ['Id'])

train,test = train_test_split(df, test_size = 0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
train_y=train.Species# output of our training data
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_y =test.Species 

model1 = svm.SVC() #select the algorithm
model1.fit(train_X,train_y) # we train the algorithm with the training data and the training output



pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,4]]))