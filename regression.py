import pandas as pd
import quandl

def main():
    print('Hi there, first regression approach')
    df = quandl.get('WIKI/GOOGL')
    print(df.head())

    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    print('Selected data')
    print(df.head())

    df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
    df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
    print('Calculated data')
    df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]
    print(df.head()) 

if __name__ == "__main__":
    main()