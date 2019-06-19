import pandas as pd
import math
import quandl # gives data form stock (historical)

def main():
    print('Hi there, first regression approach')
    df = quandl.get('WIKI/GOOGL') #we get google data
    print(df.head()) #print row data

    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    print('Selected data')#what we are interesting are some chosen
    print(df.head())

    df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
    df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
    print('Calculated data')# and those calculated are the most important
    df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]
    print(df.head()) 

    forecast_column = 'Adj. Close'
    # With many machine learning classifiers, this will just be recognized and treated as an outlier feature
    df.fillna(value=-99999, inplace=True) #NaN data with -99999 - NaN (Not a Number) cannot be used, so its substitution
    # we want to predict 10% of data-frame
    forecast_out = int(math.ceil(0.01 * len(df))) # math.ceil - map result to the nearest up

    

    df['label'] = df[forecast_column].shift(-forecast_out)

    print(df.head()) #the most recent values


    df['label'] = df[forecast_column].shift(forecast_out)
    df.dropna(inplace=True) # dropna - drop null values
    print(df.tail()) #the most recent values

if __name__ == "__main__":
    main()