# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:02:23 2021

@author: kaan_
"""

import pandas as pd

url = 'http://bilkav.com/satislar.csv'

veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:, 0:1]
y = veriler[:,1]

split = 0.33

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = split)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)
print(lr.predict(X_test), '\n')

import pickle 

dosya = "model.save"

pickle.dump(lr,open(dosya,'wb'))

yuklenen = pickle.load(open(dosya,'rb'))
print('Kaydedilmis dosyanin yuklenmesi:')
print(yuklenen.predict(X_test))



'''
import pandas as pd

url = 'http://bilkav.com/satislar.csv'

veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:, 0:1]
y = veriler[:,1]

split = 0.33

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = split)

import pickle  

yuklenen = pickle.load(open('model.save','rb'))
print(yuklenen.predict(X_test))
'''

