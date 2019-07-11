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

    # First we change labels (string names into the numbers)
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,0] = labelencoder_X.fit_transform(X[:,0])
    print(X)

    #then we introduce as many columns as many label was in first operation
    from sklearn.preprocessing import OneHotEncoder
    # onehotencoder = OneHotEncoder(categorical_features=[0])#category is in first row
    # X= onehotencoder.fit_transform(X).toarray()
    #above is depricated 
    onehotencoder = OneHotEncoder(categories='auto')
    X = np.concatenate((onehotencoder.fit_transform(X[:,0].reshape(-1,1)).toarray(),X[:,1:3]), axis=1)

    # The same possible with column transform
    # ct = ColumnTransformer(
    #     [("one_hot_encoder", OneHotEncoder(), [0])],
    #     remainder='passthrough'
    # )
    # ct.fit_transform(X)
    # X = np.array(ct.fit_transform(X))
    print(X)
        
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    print(y)


def main():
    # simpleReaderDataFrame()
    simpleReaderArray()

if __name__ == "__main__":
    main()