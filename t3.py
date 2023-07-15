import os
os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning\data\600519\20230320')
os.listdir(".")


import pandas as pd
import numpy as np

# Load csv files into pandas dataframes
orders_df = pd.read_csv('order.csv', delimiter=',')
trades_df = pd.read_csv('trade.csv', delimiter=',')

# Add necessary columns
orders_df['EventType'] = orders_df['OrderCategory'].map({'A': 'New Order', 'D': 'Cancel Order'})
trades_df['EventType'] = 'Trade Execution'
orders_df['Timestamp'] = pd.to_datetime(orders_df['TradingTime'], format='%Y%m%d%H%M%S%f')
trades_df['Timestamp'] = pd.to_datetime(trades_df['TradingTime'], format='%Y%m%d%H%M%S%f')

# Combine dataframes and sort by timestamp
combined_df = pd.concat([orders_df, trades_df])
combined_df.sort_values(by='Timestamp', inplace=True)

# Initialize order book data structure
order_book = {'bid': [], 'ask': []}

# Initialize output dataframe
columns = ['SecurityID', 'Timestamp', 'EventType'] + [f'bid_price_{i}' for i in range(1, 11)] + [f'ask_price_{i}' for i in range(1, 11)] + [f'bid_quantity_{i}' for i in range(1, 11)] + [f'ask_quantity_{i}' for i in range(1, 11)]
output_df = pd.DataFrame(columns=columns)

# Iterate over sorted dataframe
for idx, row in combined_df.iterrows():
    side = 'bid' if row['TradeMark'] == 'B' else 'ask'
    #print(side, row['TradeMark'], row)
    # Update order book
    if row['EventType'] == 'New Order':
        order_book[side].append((row['CommissionPx'], row['COmmissionAmount']))
    elif row['EventType'] == 'Cancel Order' and (row['CommissionPx'], row['COmmissionAmount']) in order_book[side]:
        order_book[side].remove((row['CommissionPx'], row['COmmissionAmount']))
    elif row['EventType'] == 'Trade Execution':
        for i, (price, quantity) in enumerate(order_book[side]):
            if price == row['TradePx']:
                order_book[side][i] = (price, quantity - row['TradeAmount'])
                if quantity - row['TradeAmount'] <= 0 and (price, quantity) in order_book[side]:
                    order_book[side].remove((price, quantity))
                break

    # Sort order book
    order_book['bid'] = sorted(order_book['bid'], key=lambda x: x[0], reverse=True)
    order_book['ask'] = sorted(order_book['ask'], key=lambda x: x[0])


    tmp_bid = order_book['bid'] + [(0, 0)] * (10 - len(order_book['bid'] ))
    tmp_ask = order_book['ask'] + [(0, 0)] * (10 - len(order_book['ask'] ))
    # Output snapshot of order book
    #print(order_book)
    snapshot = [row['SecurityID'], row['Timestamp'], row['EventType']] + [price for price, quantity in tmp_bid[:10]] + [price for price, quantity in tmp_ask[:10]] + [quantity for price, quantity in tmp_bid[:10]] + [quantity for price, quantity in tmp_ask[:10]]
    print(snapshot)
    output_df.loc[len(output_df)] = snapshot

# Write output dataframe to csv file
output_df.to_csv('output.csv', index=False)