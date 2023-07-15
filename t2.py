import os
from datetime import datetime, time

import pandas as pd

os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning\data\600519\20230320')
os.listdir(".")


import csv
from decimal import Decimal
from collections import defaultdict

# Define constants for column indices
SECURITY_ID = 0
TIMESTAMP = 1
PRICE = 2
QUANTITY = 3
SIDE = 5
ACTION = 6
ORDER_NO = 7
CHANNEL_NO = 8
SERIAL_NO = 9
SENDING_TIME = 11

# Define constants for output column names
OUTPUT_COLUMNS = ['SecurityID', 'timestamp'] + \
                 [f'bid_price_{i}' for i in range(1, 11)] + \
                 [f'ask_price_{i}' for i in range(1, 11)] + \
                 [f'bid_quantity_{i}' for i in range(1, 11)] + \
                 [f'ask_quantity_{i}' for i in range(1, 11)]
start_time = time(9, 30, 0)
end_time = time(14, 57, 0)

# Check if the time component of dt is within the time range


# Define a helper function to parse timestamps
def parse_timestamp(ts_str):
    #return datetime.strptime(ts_str, '%Y%m%d%H%M%S%f').timestamp()
    return datetime.strptime(ts_str, '%Y%m%d%H%M%S%f')

# Define a data structure to represent the order book
class OrderBook:
    def __init__(self):
        self.bids = defaultdict(Decimal)  # Maps price levels to total bid quantity
        self.asks = defaultdict(Decimal)  # Maps price levels to total ask quantity
        self.bid_counts = defaultdict(int)  # Maps price levels to number of bids
        self.ask_counts = defaultdict(int)  # Maps price levels to number of asks

    # Add a new order to the order book
    def add_order(self, side, price, quantity):
        if side == 'B':
            self.bids[price] += quantity
            self.bid_counts[price] += 1
        elif side == 'S':
            self.asks[price] += quantity
            self.ask_counts[price] += 1
        else:
            print(side)

    # Cancel an order in the order book
    def cancel_order(self, side, price, quantity):
        if side == 'B':
            self.bids[price] -= quantity
            if self.bids[price] <= 0:
                del self.bids[price]
                del self.bid_counts[price]
        elif side == 'A':
            self.asks[price] -= quantity
            if self.asks[price] <= 0:
                del self.asks[price]
                del self.ask_counts[price]

    # Execute a trade in the order book
    def execute_trade(self, side, price, quantity):
        if side == 'B':
            while quantity > 0 and len(self.asks) > 0:
                best_ask = min(self.asks.keys())
                if price >= best_ask:
                    # Match the trade with the best ask
                    traded_qty = min(quantity, self.asks[best_ask])
                    self.asks[best_ask] -= traded_qty
                    if self.asks[best_ask] <= 0:
                        del self.asks[best_ask]
                        del self.ask_counts[best_ask]
                    quantity -= traded_qty
                else:
                    break
        elif side == 'A':
            while quantity > 0 and len(self.bids) > 0:
                best_bid = max(self.bids.keys())
                if price <= best_bid:
                    # Match the trade with the best bid
                    traded_qty = min(quantity, self.bids[best_bid])
                    self.bids[best_bid] -= traded_qty
                    if self.bids[best_bid] <= 0:
                        del self.bids[best_bid]
                        del self.bid_counts[best_bid]
                    quantity -= traded_qty
                else:
                    break

    # Get the top bid and ask prices and quantities
    def get_top_levels(self):
        bid_prices = sorted(self.bids.keys(), reverse=True)
        ask_prices = sorted(self.asks.keys())
        bid_quantities = [self.bids[price] for price in bid_prices]
        ask_quantities = [self.asks[price] for price in ask_prices]
        return bid_prices[:10], ask_prices[:10], bid_quantities[:10], ask_quantities[:10]


# Define a function to process the order.csv file and update the order book accordingly
def process_order_csv(filename, order_book):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            security_id = row[0]
            timestamp = parse_timestamp(row[1])
            if start_time <= timestamp.time() <= end_time:
                action = row[6]
                side = row[5]
                price = Decimal(row[3])
                quantity = Decimal(row[4])
                order_no = row[2]
                channel_no = row[8]
                serial_no = row[9]
                if action == 'A':
                    order_book.add_order(side, price, quantity)
                elif action == 'D':
                    order_book.cancel_order(side, price, quantity)
                yield (security_id, timestamp, order_book.get_top_levels())


# Define a function to process the trade.csv file and update the order book accordingly
def process_trade_csv(filename, order_book):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            security_id = row[0]
            timestamp = parse_timestamp(row[1])
            if start_time <= timestamp.time() <= end_time:
                price = Decimal(row[2])
                quantity = Decimal(row[3])
                order_no = row[5]
                offer_no = row[6]
                trade_no = row[7]
                channel_code = row[8]
                sb_mark = row[9]
                serial_no = row[10]
                sending_time = row[11]
                side = 'B' if offer_no == 'N' else 'A'
                order_book.execute_trade(side, price, quantity)
                yield (security_id, timestamp, order_book.get_top_levels())


# Define a function to merge events from the two files and output the order book snapshot at each event
def build_order_book_snapshot(order_csv, trade_csv, output_csv):
    order_book = OrderBook()
    events = sorted(list(process_order_csv(order_csv, order_book)) + list(process_trade_csv(trade_csv, order_book)),
                    key=lambda e: e[1])
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for event in events:
            security_id, timestamp, (bid_prices, ask_prices, bid_quantities, ask_quantities) = event
            row = [security_id, timestamp] + \
                  [float(bid_prices[i]) if i < len(bid_prices) else None for i in range(10)] + \
                  [float(ask_prices[i]) if i < len(ask_prices) else None for i in range(10)] + \
                  [float(bid_quantities[i]) if i < len(bid_quantities) else None for i in range(10)] + \
                  [float(ask_quantities[i]) if i < len(ask_quantities) else None for i in range(10)]
            writer.writerow(row)


build_order_book_snapshot('order.csv', 'trade.csv', 'output.csv')

import os
from datetime import datetime, time

import pandas as pd

os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning\data\600519\20230320')
os.listdir(".")

import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('order_book.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
filtered_df = df[df['timestamp'].dt.time.between(time(9, 30), time(14, 57))]
filtered_df.plot(x='timestamp', y=['bid_price_1', 'ask_price_1'])
#filtered_df.plot(x='timestamp', y=['ask_price_1'])
plt.show()

df = pd.read_csv(r'c:\work\china-data\MarketMakingViaReinforcementLearning\order_book.remote.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
filtered_df = df[df['timestamp'].dt.time.between(time(9, 30), time(14, 57))]
filtered_df.plot(x='timestamp', y=['bid_price_1'])
filtered_df.plot(x='timestamp', y=['ask_price_1'])
plt.show()


df['bid_price_1'].describe()
df['ask_price_1'].describe()

order_book_df[:500].plot(x='timestamp', y=['bid_price_1'])
plt.show()

(order_book_df['bid_price_1'] == 1900).value_counts()
(order_book_df['ask_price_1'] == 0).value_counts()

filtered_df['bid_price_1'].describe()

order_book_df['bid_price_1']