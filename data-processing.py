import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer #scilearn - imputer replace data

# Taking care of missing data ...... the NEW way w/ ColumnTransformer & OneHotEncoder
    
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
def simpleReaderDataFrame():
    print('simple reader in action')
    dataset = pd.read_csv('data1.csv')
    X = dataset.iloc[:,:-1] #X is a matrix of features
    y = dataset.iloc[:,3]
    print(X)
    print(y)

    
    #replacing missing data
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
    imputer = imputer.fit(X.iloc[:, 1:3])
    X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

    print(X)
    print(y)

def simpleReaderArray():
    print('simple reader in action')
    dataset = pd.read_csv('data1.csv')
    X = dataset.iloc[:,:-1].values #X is a matrix of features - values change it to array
    y = dataset.iloc[:,3].values
    print(X)
    print(y)

    
    #replacing missing data
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])

    print(X)
    print(y)

    
    # ct = ColumnTransformer(
    #         [('ohe', OneHotEncoder(), [0])],      # use [] around the transformer to give the 3 attributes to the transformer
    #                                                 # name = 'ohe', transformer = OneHotEncoder(), [0] = first column
    #         remainder = 'passthrough')            # passes the rest of the columns through untouched
    # X = np.array(ct.fit_transform(X), dtype = np.float) # make the first column of X = the transformed column

    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,0] = labelencoder_X.fit_transform(X[:,0])
    print(X)


def main():
    # simpleReaderDataFrame()
    simpleReaderArray()

if __name__ == "__main__":
    main()