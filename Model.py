import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("Social_Network_Ads.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler()
train_X=sc.fit_transform(train_X)
test_X=sc.transform(test_X)
cls=LogisticRegression()
cls.fit(train_X,train_y)
pred_y=cls.predict(test_X)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,pred_y))
print(cls.predict(sc.transform([[40,90000]])))
a=[cls,sc]
with open("lucky.pkl","wb") as f:
    pickle.dump(a,f)
