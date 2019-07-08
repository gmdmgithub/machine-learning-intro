import pandas as pd
import math
import quandl  # gives data form stock (historical)
import numpy as np  # arrays in pyton
# preprocessing - scaling data, #svm - support vector machine - do regretion
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
# cross_validate - create training and testing,
from sklearn import model_selection


def lineBreaker(text):
    print("############### "+text+"########################\n")


def main():
    print('Hi there, first regression approach')
    df = quandl.get('WIKI/GOOGL')  # we get google data
    print(df.head())  # print row data

    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    print('Selected data')  # what we are interesting are some chosen
    print(df.head())

    df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
    df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
    print('Calculated data')  # and those calculated are the most important
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    print(df.head())

    forecast_column = 'Adj. Close'
    # With many machine learning classifiers, this will just be recognized and treated as an outlier feature
    # NaN data with -99999 - NaN (Not a Number) cannot be used, so its substitution
    df.fillna(value=-99999, inplace=True)
    # we want to predict 10% of data-frame
    # math.ceil - map result to the nearest up
    forecast_out = int(math.ceil(0.01 * len(df)))

    # for head data
    df['label'] = df[forecast_column].shift(-forecast_out)
    print(df.head())  # the most recent values

    # for tail data
    df['label'] = df[forecast_column].shift(forecast_out)
    df.dropna(inplace=True)  # dropna - drop null values
    print(df.tail())  # the most recent values

    lineBreaker("training and testing")
    # X - capital x - features, y - labels
    X = np.array(df.drop(['label'], 1))  # all data except label
    y = np.array(df['label'])
    # now we scale X
    X = preprocessing.scale(X)  # it normalize data
    y = np.array(df['label'])

    print(len(X), len(y))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)  # fit - train
    accuracy = clf.score(X_test, y_test)  # score - test

    print(accuracy)


if __name__ == "__main__":
    main()
