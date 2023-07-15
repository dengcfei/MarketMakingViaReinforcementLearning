
import pandas as pd
import numpy as np
df = pd.read_csv(r'data/600519/20230320/order_book.csv')

df = df.rename(columns =
                             {
                                 'bid_price_1' : 'bid1_price',
                                 'ask_price_1' : 'ask1_price',
                             })

df_price = df[['SecurityID','timestamp','bid1_price','ask1_price']]
df_price['midprice'] = 0.5 * (df['bid1_price'] + df['ask1_price'])
df_price['spread'] =  (df['ask1_price'] - df['bid1_price'])
df_price.to_csv(r'data/600519/20230320/price.csv', index=False)
