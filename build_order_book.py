import os
from datetime import datetime, time

import pandas as pd



import csv
from decimal import Decimal
from collections import defaultdict
from dataclasses import asdict, dataclass, astuple

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
start_time = time(9, 15, 0)
end_time = time(15, 0, 0)

# Check if the time component of dt is within the time range


# Define a helper function to parse timestamps
def parse_timestamp(ts_str):
    #return datetime.strptime(ts_str, '%Y%m%d%H%M%S%f').timestamp()
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')

@dataclass
class Order:
    # Experiment
    id: int = -1
    side: str = "C"
    price: Decimal = 0.0
    quantity: Decimal = 0.0
    trading_time: datetime = datetime(1900, 1, 1, 0, 0, 0)
@dataclass
class Trade:
    # Experiment
    bid_id: int = -1
    ask_id: int = -1
    price: Decimal = 0.0
    quantity: Decimal = 0.0
    sb_mark: str = 'C'
    trading_time: datetime = datetime(1900, 1, 1, 0, 0, 0)

@dataclass
class Statistics:
    limit_buy_n: int = 0
    limit_buy_volume: int = 0
    market_buy_n: int = 0
    market_buy_volume: int = 0
    withdraw_buy_n: int = 0
    withdraw_buy_volume: int = 0
    limit_sell_n: int = 0
    limit_sell_volume: int = 0
    market_sell_n: int = 0
    market_sell_volume: int = 0
    withdraw_sell_n: int = 0
    withdraw_sell_volume: int = 0

# Define a data structure to represent the order book
class OrderBook:
    def __init__(self):
        self.bids = defaultdict(Decimal)  # Maps price levels to total bid quantity
        self.asks = defaultdict(Decimal)  # Maps price levels to total ask quantity
        #self.bid_counts = defaultdict(int)  # Maps price levels to number of bids
        #self.ask_counts = defaultdict(int)  # Maps price levels to number of asks
        self.bid_orders = defaultdict(dict)  # Maps price levels to total bid quantity
        self.ask_orders = defaultdict(dict)  # Maps price levels to total ask quantity
        self.statistics = defaultdict(dict)  # Maps price levels to total ask quantity

    # Add a new order to the order book
    def add_order(self, order: Order):
        #if order.id == '9708393':
        #    print("found")



        if order.side == 'B':

            self.bids[order.price] += order.quantity
            #self.bid_counts[order.price] += 1
            self.bid_orders[order.id] = order
            #print("add_order, buy ", order)
        elif order.side == 'S':
            self.asks[order.price] += order.quantity
            #self.ask_counts[order.price] += 1
            self.ask_orders[order.id] = order
            #print("add_order, sell ", order)
        else:
            print("unknown side", order)

    # Cancel an order in the order book
    def cancel_order(self, order: Order):
        if order.trading_time in self.statistics.keys():
            stat = self.statistics[order.trading_time]
        else:
            stat = Statistics()
            self.statistics[order.trading_time] = stat

        #if order.id == '9708393':
        #    print("found")
        if order.side == 'B':
            if order.id in self.bid_orders.keys():
                stat.withdraw_buy_n += 1
                stat.withdraw_buy_volume += order.quantity

                self.bids[order.price] -= order.quantity
                del self.bid_orders[order.id]
                #print("cancel bid order ", order)
                if self.bids[order.price] <= 0:
                    del self.bids[order.price]
                    #del self.bid_counts[order.price]
            else:
                print("cancel_order: unknown order ", order)
        elif order.side == 'S':
            if order.id in self.ask_orders.keys():
                stat.withdraw_sell_n += 1
                stat.withdraw_sell_volume += order.quantity

                del self.ask_orders[order.id]
                #print("cancel ask order ", order)
                self.asks[order.price] -= order.quantity
                if self.asks[order.price] <= 0:
                    del self.asks[order.price]
                    #del self.ask_counts[order.price]
            else:
                print("cancel_order: unknown order ", order)
        else:
            print("unknown side", order)

    # Execute a trade in the order book
    def execute_trade(self, trade: Trade):
        #if trade.ask_id == '9708393' or trade.bid_id == '9708393':
        #    print("found")
        #buy order could be mkt order and not in order book, but sell side must be present
        if trade.trading_time in self.statistics.keys():
            stat = self.statistics[trade.trading_time]
        else:
            stat = Statistics()
            self.statistics[trade.trading_time] = stat

        if trade.sb_mark == 'B':
            if trade.ask_id in self.ask_orders.keys():
                stat.limit_sell_n += 1
                stat.limit_sell_volume += trade.quantity

                self.asks[trade.price] -= trade.quantity
                self.ask_orders[trade.ask_id].quantity -= trade.quantity
                if self.ask_orders[trade.ask_id].quantity <= 0:
                    #print("del ask order ", self.ask_orders[trade.ask_id])
                    del self.ask_orders[trade.ask_id]

                if self.asks[trade.price] <= 0:
                    del self.asks[trade.price]
                    # del self.ask_counts[trade.price]
            else:
                print("unknown ask order id for trade ", trade.ask_id, trade)
                # even though cannot find order, still update quantity
                self.asks[trade.price] -= trade.quantity
                if self.asks[trade.price] <= 0:
                    del self.asks[trade.price]

            if trade.bid_id in self.bid_orders.keys():
                if self.bid_orders[trade.bid_id].trading_time == trade.trading_time:
                    stat.market_buy_n += 1
                    stat.market_buy_volume += trade.quantity
                else:
                    stat.limit_buy_n += 1
                    stat.limit_buy_volume += trade.quantity

                    self.bids[trade.price] -= trade.quantity
                    self.bid_orders[trade.bid_id].quantity -= trade.quantity
                    if self.bid_orders[trade.bid_id].quantity <= 0:
                        #print("del bid order ", self.bid_orders[trade.bid_id])
                        del self.bid_orders[trade.bid_id]
                    if self.bids[trade.price] <= 0:
                        del self.bids[trade.price]
                        #del self.bid_counts[trade.price]
            else:
                pass #print("mkt bid order id for trade ", trade.bid_id, trade)
        elif trade.sb_mark == 'S':
            if trade.bid_id in self.bid_orders.keys():
                stat.limit_buy_n += 1
                stat.limit_buy_volume += trade.quantity

                self.bids[trade.price] -= trade.quantity
                self.bid_orders[trade.bid_id].quantity -= trade.quantity
                if self.bid_orders[trade.bid_id].quantity <= 0:
                    #print("del bid order ", self.bid_orders[trade.bid_id])
                    del self.bid_orders[trade.bid_id]
                if self.bids[trade.price] <= 0:
                    del self.bids[trade.price]
                    #del self.bid_counts[trade.price]
            else:
                print("unknown bid order id for trade ", trade.bid_id, trade)
                #even though cannot find order, still update quantity
                self.bids[trade.price] -= trade.quantity
                if self.bids[trade.price] <= 0:
                    del self.bids[trade.price]


            if trade.ask_id in self.ask_orders.keys():
                if self.ask_orders[trade.ask_id].trading_time == trade.trading_time:
                    stat.market_sell_n += 1
                    stat.market_sell_volume += trade.quantity
                else:
                    stat.limit_sell_n += 1
                    stat.limit_sell_volume += trade.quantity

                    self.asks[trade.price] -= trade.quantity
                    self.ask_orders[trade.ask_id].quantity -= trade.quantity
                    if self.ask_orders[trade.ask_id].quantity <= 0:
                        #print("del ask order ", self.ask_orders[trade.ask_id])
                        del self.ask_orders[trade.ask_id]
                    if self.asks[trade.price] <= 0:
                        del self.asks[trade.price]
                        # del self.ask_counts[trade.price]
            else:
                pass #print("mkt ask order id for trade ", trade.ask_id, trade)
        elif trade.sb_mark == 'N':
            if trade.bid_id in self.bid_orders.keys():
                order = self.bid_orders[trade.bid_id]
                if order.price >= trade.price:
                    self.bids[order.price] -= trade.quantity
                    self.bid_orders[trade.bid_id].quantity -= trade.quantity
                    if self.bid_orders[trade.bid_id].quantity <= 0:
                        #print("del bid order ", self.bid_orders[trade.bid_id])
                        del self.bid_orders[trade.bid_id]
                    if self.bids[order.price] <= 0:
                        del self.bids[order.price]
            if trade.ask_id in self.ask_orders.keys():
                order = self.ask_orders[trade.ask_id]
                if order.price <= trade.price:
                    self.asks[order.price] -= trade.quantity
                    self.ask_orders[trade.ask_id].quantity -= trade.quantity
                    if self.ask_orders[trade.ask_id].quantity <= 0:
                        #print("del ask order ", self.ask_orders[trade.ask_id])
                        del self.ask_orders[trade.ask_id]
                    if self.asks[order.price] <= 0:
                        del self.asks[order.price]
                    # del self.ask_counts[trade.price]


    # Get the top bid and ask prices and quantities
    def get_top_levels(self):
        bid_prices = sorted(self.bids.keys(), reverse=True)
        ask_prices = sorted(self.asks.keys())
        bid_quantities = [self.bids[price] for price in bid_prices]
        ask_quantities = [self.asks[price] for price in ask_prices]
        #if float(bid_prices[0]) == 1752.92:
        #    print("found 1752.92", self.bid_orders)
        return bid_prices[:10], ask_prices[:10], bid_quantities[:10], ask_quantities[:10]


#['SecurityID','CommissionTime','CommissionOrderNo','CommissionPx','COmmissionAmount','TradeMark','OrderCategory','CommissionNo','ChannelNo','SerialNo']

def process_order_event(row, order_book):
    security_id = row['SecurityID']
    timestamp = parse_timestamp(str(row['TradingTime']))
    if start_time <= timestamp.time() <= end_time:
        action = row['OrderCategory']
        side = row['TradeMark']
        price = Decimal(row['CommissionPx'])
        quantity = Decimal(row['COmmissionAmount'])
        order_no = row['CommissionOrderNo']
        channel_no = row['ChannelNo']
        serial_no = row['SerialNo']
        if action == 'A':
            order_book.add_order(Order(order_no, side, price, quantity, timestamp))
        elif action == 'D':
            order_book.cancel_order(Order(order_no, side, price, quantity, timestamp))
        return (security_id, timestamp, order_book.get_top_levels())


# ['SecurityID','TradeTime','TradePx','TradeAmount','TradeValue','OrderNo','OfferNo','TradeNo','ChannelCode','SBMark','SerialNo','SendingTime']

def process_trade_event(row, order_book):
    security_id = row['SecurityID']
    timestamp = parse_timestamp(str(row['TradingTime']))
    if start_time <= timestamp.time() <= end_time:
        price = Decimal(row['TradePrice'])
        quantity = Decimal(row['TradeVolume'])
        bid_no = row['OrderNo']
        offer_no = row['OfferNo']
        trade_no = row['TradeNo']
        channel_code = row['ChannelCode']
        # N, B, S
        sb_mark = row['SBMark']
        serial_no = row['SerialNo']
        sending_time = row['SendingTime']
        order_book.execute_trade(Trade(bid_no, offer_no, price, quantity, sb_mark, timestamp))
        return (security_id, timestamp, order_book.get_top_levels())


# Define a function to merge events from the two files and output the order book snapshot at each event
def build_order_book_snapshot(order_csv, trade_csv, output_csv, msg_csv):
    order_book = OrderBook()

    orders_df = pd.read_csv(order_csv, delimiter=',', dtype=str)
    trades_df = pd.read_csv(trade_csv, delimiter=',', dtype=str)

    orders_df['EventType'] = 'OrderEvent'
    trades_df['EventType'] = 'TradeEvent'
    orders_df['Timestamp'] = pd.to_datetime(orders_df['TradingTime'])
    trades_df['Timestamp'] = pd.to_datetime(trades_df['TradingTime'])
    combined_df = pd.concat([orders_df, trades_df])
    combined_df.sort_values(by=['Timestamp', 'EventType'], inplace=True)

    events = []
    for idx, row in combined_df.iterrows():
        if row['EventType'] == 'OrderEvent':
            ret = process_order_event(row, order_book)
            if ret:
                events.append(ret)
        elif row['EventType'] == 'TradeEvent':
            ret = process_trade_event(row, order_book)
            if ret:
                events.append(ret)

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
    with open(msg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        first_stat = list(order_book.statistics.values())[0]
        writer.writerow(['timestamp'] + list(first_stat.__dataclass_fields__.keys()))
        for key, value in order_book.statistics.items():
            row = [key.strftime("%Y-%m-%d %H:%M:%S.%f")] + list(astuple(value))
            writer.writerow(row)


date_str='20230320'
build_order_book_snapshot(f'data/600519/{date_str}/order.csv', f'data/600519/{date_str}/trade.csv', f'data/600519/{date_str}/order_book.csv', f'data/600519/{date_str}/msg.csv')
date_str='20230321'
build_order_book_snapshot(f'data/600519/{date_str}/order.csv', f'data/600519/{date_str}/trade.csv', f'data/600519/{date_str}/order_book.csv', f'data/600519/{date_str}/msg.csv')
date_str='20230322'
build_order_book_snapshot(f'data/600519/{date_str}/order.csv', f'data/600519/{date_str}/trade.csv', f'data/600519/{date_str}/order_book.csv', f'data/600519/{date_str}/msg.csv')
