import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json


# streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")
# df = pd.DataFrame(streeteasy)
# x = df[['bedrooms','bathrooms','has_gym','has_patio','size_sqft']]
# y = df[['rent']]
# X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, random_state = 1)
# model = LinearRegression()
# model.fit(X_train,y_train)
# y_predict = model.predict(X_test)
# new_test = [[2,1,0,0,500]]
# new_predict = model.predict(new_test)

# print(new_predict)


with open('data.json','r') as file:
    data = json.load(file)
random_list = []
for i in data:
    f = data[i]
    for o in f:
        keys,values = zip(*o.items())
        random_list.append(list(values))

    
# municipal = pd.read_json('data.json')
# print(municipal)

columns = ['REGION_EN','LOCATION_EN','ULTIMATE_RECIPIENT_EN','ORG_TP_DSC_EN','2005-2006','2006-2007',
           '2007-2008','2008-2009','2009-2010','2010-2011','2011-2012','2013-2014','2014-2015','2016-2017',
           '2017-2018','2018-2019','2019-2020','2020-2021','2021-2022','2022-2023']





