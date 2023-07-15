import os
os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning\data\600519\20230320')
os.listdir(".")

import csv
from collections import defaultdict
from datetime import datetime

# Define the order book data structure
order_book = defaultdict(lambda: {'bids': [], 'asks': []})

# Define the output file name
output_file = 'output.csv'

# Read the order.csv file and build the order book
with open('order.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        security_id = row['SecurityID']
        price = float(row['CommissionPx'])
        qty = int(row['COmmissionAmount'])
        order_no = row['CommissionOrderNo']
        order_category = row['OrderCategory']
        if order_category == 'A':
            # Add a new order
            timestamp = datetime.strptime(row['TradingTime'], '%Y%m%d%H%M%S%f')
            order_book[security_id]['asks'].append((price, qty, order_no, timestamp))
        elif order_category == 'D':
            # Cancel an order
            for i, (p, q, n, t) in enumerate(order_book[security_id]['asks']):
                if n == order_no:
                    del order_book[security_id]['asks'][i]
                    break

# Read the trade.csv file and update the order book
with open('trade.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        security_id = row['SecurityID']
        price = float(row['TradePx'])
        qty = int(row['TradeAmount'])
        timestamp = datetime.strptime(row['TradingTime'], '%Y%m%d%H%M%S%f')
        # Match trades with ask orders
        for i, (p, q, n, t) in enumerate(order_book[security_id]['asks']):
            if price >= p:
                if qty >= q:
                    # Execute the entire order
                    qty -= q
                    del order_book[security_id]['asks'][i]
                else:
                    # Partially execute the order
                    order_book[security_id]['asks'][i] = (p, q - qty, n, t)
                    qty = 0
                if qty == 0:
                    break
        # Match trades with bid orders
        for i, (p, q, n, t) in enumerate(order_book[security_id]['bids']):
            if price <= p:
                if qty >= q:
                    # Execute the entire order
                    qty -= q
                    del order_book[security_id]['bids'][i]
                else:
                    # Partially execute the order
                    order_book[security_id]['bids'][i] = (p, q - qty, n, t)
                    qty = 0
                if qty == 0:
                    break

# Define the output field names
field_names = ['SecurityID', 'timestamp']
for i in range(1, 11):
    field_names += [f'bid_price_{i}', f'bid_quantity_{i}', f'bid_count_{i}']
for i in range(1, 11):
    field_names += [f'ask_price_{i}', f'ask_quantity_{i}', f'ask_count_{i}']

# Write the output file
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    timestamps = set()
    for security_id, orders in order_book.items():
        for p, q, n, t in orders['bids']:
            timestamps.add(t)
        for p, q, n, t in orders['asks']:
            timestamps.add(t)
    timestamps = sorted(timestamps)
    for timestamp in timestamps:
        row = {'SecurityID': '', 'timestamp': timestamp.strftime('%Y%m%d%H%M%S%f')}
        for i in range(1, 11):
            if i <= len(order_book[security_id]['bids']):
                p, q, n, t = order_book[security_id]['bids'][-i]
                row[f'bid_price_{i}'] = str(p)
                row[f'bid_quantity_{i}'] = str(q)
                row[f'bid_count_{i}'] = str(len(order_book[security_id]['bids']))
            else:
                row[f'bid_price_{i}'] = ''
                row[f'bid_quantity_{i}'] = ''
                row[f'bid_count_{i}'] = ''
        for i in range(1, 11):
            if i <= len(order_book[security_id]['asks']):
                p, q, n, t = order_book[security_id]['asks'][i - 1]
               row[f'ask_price_{i}'] = str(p)
                row[f'ask_quantity_{i}'] = str(q)
                row[f'ask_count_{i}'] = str(len(order_book[security_id]['asks']))
            else:
                row[f'ask_price_{i}'] = ''
                row[f'ask_quantity_{i}'] = ''
                row[f'ask_count_{i}'] = ''
        writer.writerow(row)