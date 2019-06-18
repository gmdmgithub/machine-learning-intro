import pandas as pd
import quandl

def main():
    print('Hi there, first regression approach')
    df = quandl.get('WIKI/GOOGL')
    print(df.head()) 

if __name__ == "__main__":
    main()