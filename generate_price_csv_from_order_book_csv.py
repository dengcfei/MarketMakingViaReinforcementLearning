
import pandas as pd
import numpy as np

def process_one_price_csv(date_str):

    df = pd.read_csv(f'data/600519/{date_str}/order_book.csv')
    df = df.rename(columns =
                                 {
                                     'bid_price_1' : 'bid1_price',
                                     'ask_price_1' : 'ask1_price',
                                 })

    df_price = df[['SecurityID','timestamp','bid1_price','ask1_price']]
    df_price['midprice'] = 0.5 * (df['bid1_price'] + df['ask1_price'])
    df_price['spread'] =  (df['ask1_price'] - df['bid1_price'])
    df_price.to_csv(f'data/600519/{date_str}/price.csv', index=False)

for date_str in [
        '20230320',
        '20230321',
        '20230322',
        '20230323',
        '20230324',
        '20230327',
        '20230328',
        '20230329',
        '20230330',
        '20230331']:
    print("processing ", date_str)
    process_one_price_csv(date_str)
