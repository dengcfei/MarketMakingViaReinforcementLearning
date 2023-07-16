
import pandas as pd
import numpy as np
date_str='20230322'
df = pd.read_csv(f'data/600519/{date_str}/order_book.csv')

df = df.rename(columns =
                             {
                                 'bid_price_1' : 'bid1_price',
                                 'ask_price_1' : 'ask1_price',
                             })

df_price = df[['SecurityID','timestamp','bid1_price','ask1_price']]
#df_price = df_price.drop_duplicates(subset='timestamp', keep='last')
df_price['midprice'] = 0.5 * (df['bid1_price'] + df['ask1_price'])
df_price['spread'] =  (df['ask1_price'] - df['bid1_price'])
df_price.to_csv(f'data/600519/{date_str}/price.csv', index=False)
