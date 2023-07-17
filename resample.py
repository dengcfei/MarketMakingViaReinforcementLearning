import os
import pandas as pd

date_str = '20230322'
df_order = pd.read_csv(f'data/600519/{date_str}/order.csv')
df_trade = pd.read_csv(f'data/600519/{date_str}/trade.csv')
df_order_book = pd.read_csv(f'data/600519/{date_str}/order_book.csv')
df_msg = pd.read_csv(f'data/600519/{date_str}/msg.csv')

df_order['TradingTime'] = pd.to_datetime(df_order['TradingTime'])
df_order.set_index('TradingTime', inplace=True)
df_order_1s = df_order.resample('1s').last().reset_index()

df_trade['TradingTime'] = pd.to_datetime(df_trade['TradingTime'])
df_trade.set_index('TradingTime', inplace=True)
df_agg = df_trade.resample('1s').agg({
    'TradeVolume' : 'sum',
    'TradeValue' : 'sum',
    'SecurityID' : 'last',
    'OrderNo' : 'last',
    'OfferNo' : 'last',
    'TradeNo' : 'last',
    'ChannelCode' : 'last',
    'SBMark' : 'last',
    'SerialNo' : 'last',
    'SendingTime' : 'last',
})
df_agg['TradePrice'] = df_agg['TradeValue'] / df_agg['TradeVolume']
df_agg = df_agg.reset_index()

df_order_book['timestamp'] = pd.to_datetime(df_order_book['timestamp'])
df_order_book.set_index('timestamp', inplace=True)
df_order_book_1s = df_order_book.resample('1s').last().reset_index()

df_msg['timestamp'] = pd.to_datetime(df_msg['timestamp'])
df_msg.set_index('timestamp', inplace=True)
df_msg_1s = df_msg.resample('1s').last().reset_index()

df_order_1s.dropna(inplace=True)
df_agg.dropna(inplace=True)
df_order_book_1s.dropna(inplace=True)
df_msg_1s.dropna(inplace=True)

df_order_1s.to_csv(f'data/600519/{date_str}/order_sample.csv', index=False)
df_agg.to_csv(f'data/600519/{date_str}/trade_sample.csv', index=False)
df_order_book_1s.to_csv(f'data/600519/{date_str}/order_book_sample.csv', index=False)
df_msg_1s.to_csv(f'data/600519/{date_str}/msg_sample.csv', index=False)