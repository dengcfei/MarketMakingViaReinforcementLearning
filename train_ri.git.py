#utility
import pandas as pd
import numpy as np
import math
from datetime import datetime, time
import time as time_module

def min_max_norm(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    if ranges==0:
        return data*0
    normData = data - minVals
    normData = normData/ranges
    return normData

def z_norm(data):
    return (data-data.mean())/(data.std()+1e-7)

def lob_norm(data_, midprice):
    data = data_.copy()
    for i in range(10):
        data[f'ask_price_{i+1}'] = data[f'ask_price_{i+1}']/(midprice+1e-7) - 1
        data[f'bid_price_{i+1}'] = data[f'bid_price_{i+1}']/(midprice+1e-7) - 1
        # data[f'ask{i+1}_price'] = z_norm(data[f'ask{i+1}_price'])
        # data[f'bid{i+1}_price'] = z_norm(data[f'bid{i+1}_price'])
        data[f'ask_quantity_{i+1}'] = data[f'ask_quantity_{i+1}']/data[f'ask_quantity_{i+1}'].max()
        data[f'bid_quantity_{i+1}'] = data[f'bid_quantity_{i+1}']/data[f'bid_quantity_{i+1}'].max()

    return data

def onehot_label(targets):
    from tensorflow import keras
    # targets: pd.DataFrame len(data)*n_horizons
    all_label = []
    for i in range(targets.shape[1]):
        label = targets.iloc[:,i] - 1
        label = keras.utils.to_categorical(label, 3)
        # label = label.reshape(len(label), 1, 3)
        all_label.append(label)
    return np.hstack(all_label)

def day2date(day):
    day = list(day)
    day.insert(4,'-')
    day.insert(7,'-')
    date = ''.join(day)
    return date

def pd_is_equal(state_1, state_2):
    tmp_1 = state_1.iloc[:,1:]
    tmp_2 = state_2.iloc[:,1:]
    return tmp_1.equals(tmp_2)

def load_data(code, datelist, horizon=10):
    if type(datelist) is str:
        datelist = [datelist]
    data_list = []
    for day in datelist:
        #ask = pd.read_csv(f"data/{code}/{day}/ask.csv")
        #bid = pd.read_csv(f"data/{code}/{day}/bid.csv").drop(['timestamp'], axis = 1)
        #price = pd.read_csv(f"data/{code}/{day}/price.csv").drop(['timestamp', 'ask1_price', 'bid1_price'], axis = 1)
        #data = pd.concat([ask, bid, price], axis=1)
        price = pd.read_csv(f"data/{code}/{day}/price.csv")
        data['date'] = data['timestamp'].str.split(expand=True)[0]
        data['time'] = data['timestamp'].str.split(expand=True)[1]
        data.drop('timestamp', axis=1, inplace=True)

        data['y']=getLabel(data.midprice, horizon)

        data_list.append(data)
    return pd.concat(data_list)

def getLabel(mid_price, horizon, threshold=1e-5):
    price_past = mid_price.rolling(window=horizon).mean()

    price_future = mid_price.copy()
    price_future[:-horizon] = price_past[horizon:]
    price_future[-horizon:] = np.nan

    pct_change = (price_future - price_past)/price_past
    pct_change[pct_change>=threshold] = 1
    pct_change[(pct_change<threshold) & (-threshold<pct_change)] = 2
    pct_change[pct_change<=-threshold] = 3
    return pct_change

def process_data(data):
    data = data[(data.time > '10:00:00')&(data.time < '14:30:00')]
    data = data.dropna()
    data.y = data.y.astype(int)

    for i in range(10):
        data[f'ask{i+1}_price'] = data[f'ask{i+1}_price']/data['midprice'] - 1
        data[f'bid{i+1}_price'] = data[f'bid{i+1}_price']/data['midprice'] - 1
        # data[f'ask{i+1}_price'] = z_norm(data[f'ask{i+1}_price'])
        # data[f'bid{i+1}_price'] = z_norm(data[f'bid{i+1}_price'])
        data[f'ask{i+1}_volume'] = data[f'ask{i+1}_volume']/data[f'ask{i+1}_volume'].max()
        data[f'bid{i+1}_volume'] = data[f'bid{i+1}_volume']/data[f'bid{i+1}_volume'].max()

    return data.set_index(['date', 'time'])

def reorder(data):
    '''
    reorder the data to this order:
    ask1_v, ask1_p, bid1_v, bid1_p ... ask10_v, ask10_p, bid10_v, bid10_p
    '''
    data=np.array(data)
    data=data.reshape(data.shape[0], 4, 10)
    data= np.transpose(data, (0,2,1))
    data = data.reshape(data.shape[0], -1)
    return data

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY

def price_legal_check(ask_price, bid_price):
    # legal check
    ask_price = math.ceil(100*ask_price)/100
    bid_price = math.floor(100*bid_price)/100
    return ask_price, bid_price

def getRealizedVolatility(data, resample='min'):
    if resample:
        data = data.resample(resample).last()

    midprice_lag = data.shift(1)
    midprice_log = data.apply(np.log)
    midprice_lag_log = midprice_lag.apply(np.log)
    r = midprice_log - midprice_lag_log
    r2 = r*r
    rv = r2.sum()

    return rv

def getRelativeStrengthIndex(data):
    length = len(data)
    data = data.resample('s').last()
    data = data.pct_change(1)
    gain = data[data>0].sum()/length
    loss = -data[data<0].sum()/length
    if gain or loss:
        rsi = gain/(gain+loss)
    else:
        rsi = .5
    return rsi

def getOrderStrengthIndex(data):
    '''
    data: msg
    columns:[market_buy_volume  market_buy_n  market_sell_volume  market_sell_n  limit_buy_volume  limit_buy_n  limit_sell_volume  limit_sell_n  withdraw_buy_volume  withdraw_buy_n  withdraw_sell_volume  withdraw_sell_n]
    '''
    market_volume_intensity = (data.market_buy_volume.sum() - data.market_sell_volume.sum())/(data.market_buy_volume.sum() + data.market_sell_volume.sum() + 1e-7)
    market_number_intensity = (data.market_buy_n.sum() - data.market_sell_n.sum())/(data.market_buy_n.sum() + data.market_sell_n.sum() + 1e-7)
    limit_volume_intensity = (data.limit_buy_volume.sum() - data.limit_sell_volume.sum())/(data.limit_buy_volume.sum() + data.limit_sell_volume.sum() + 1e-7)
    limit_number_intensity = (data.limit_buy_n.sum() - data.limit_sell_n.sum())/(data.limit_buy_n.sum() + data.limit_sell_n.sum() + 1e-7)
    withdraw_volume_intensity = (data.withdraw_buy_volume.sum() - data.withdraw_sell_volume.sum())/(data.withdraw_buy_volume.sum() + data.withdraw_sell_volume.sum() + 1e-7)
    withdraw_number_intensity = (data.withdraw_buy_n.sum() - data.withdraw_sell_n.sum())/(data.withdraw_buy_n.sum() + data.withdraw_sell_n.sum() + 1e-7)

    return market_volume_intensity, market_number_intensity, limit_volume_intensity, limit_number_intensity, withdraw_volume_intensity, withdraw_number_intensity

#base_env
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# from tensorforce import Environment



TRADE_UNIT = 100

class BaseEnv():
    """
    """
    def __init__(
            self,
            initial_value=0,
            max_episode_timesteps=1000,
            data_dir='./data',
            log=1,
            experiment_name='',
            **kwargs
            ):
        super().__init__()
        self.name = ''
        self.initial_value = initial_value
        self.__max_episode_timesteps__=max_episode_timesteps
        self.data_dir = data_dir
        self.log = log
        self.exp_name = experiment_name

    '''
        You need to overload these functions
    '''

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError

    def action2order(self):
        raise NotImplementedError

    def get_state_at_t(self, t):
        raise NotImplementedError

    def get_reward(self, trade_price, trade_volume):
        # Define reward function here
        reward = self.value - self.value_
        self.value_ = self.value
        return reward

    '''
        Load data
    '''

    def load_orderbook(self, code, day):
        #ask = pd.read_csv(self.data_dir + f'/{code}/{day}/ask.csv')
        #bid = pd.read_csv(self.data_dir + f'/{code}/{day}/bid.csv').drop(['timestamp'], axis = 1)

        #self.orderbook = pd.concat([ask, bid], axis=1)
        self.orderbook = pd.read_csv(f"data/{code}/{day}/order_book.csv")
        self.orderbook = self.orderbook.drop('SecurityID', axis=1)
        self.orderbook.timestamp = pd.to_datetime(self.orderbook.timestamp)
        #self.orderbook = self.orderbook[(f'{self.day} 09:30:00'<self.orderbook.timestamp)&(self.orderbook.timestamp<f'{self.day} 14:57:00')]
        self.orderbook = self.orderbook.set_index('timestamp')
        self.orderbook_length = len(self.orderbook)
        #print('load lob done!', code, day)
        #print('loaded order book', len(self.orderbook))

    def load_orderqueue(self, code, day):
        pass

    def load_price(self, code, day):
        self.price = pd.read_csv(self.data_dir + f'/{code}/{day}/price.csv')


        self.price.timestamp = pd.to_datetime(self.price.timestamp)
        self.price = self.price.set_index('timestamp')
        #print("loaded price", len(self.price))
        #self.price = self.price.loc[self.orderbook.index]
        #print("loaded price", len(self.price))

    def load_msg(self, code, day):
        self.msg = pd.read_csv(self.data_dir + f'/{code}/{day}/msg.csv')
        self.msg.timestamp = pd.to_datetime(self.msg.timestamp)
        self.msg = self.msg.set_index('timestamp')
        self.msg = self.msg.loc[self.orderbook.index]

    def load_order(self, code, day):
        #order_columns = pd.read_csv('raw/GTA_SZL2_ORDER.csv')
        #self.order = pd.read_csv(f'raw/SZL2_ORDER_{code}_{day[:6]}.csv', names=list(order_columns), low_memory=False)
        self.order = pd.read_csv(f"data/{code}/{day}/order.csv")

        self.order.TradingTime = pd.to_datetime(self.order.TradingTime)
        #self.order = self.order[self.order.TradingDate==int(day)]
        self.order = self.order[(f'{self.day} 09:30:00'<self.order.TradingTime)&(self.order.TradingTime<f'{self.day} 14:57:00')]
        #print("loaded order", len(self.order))

    def load_trade(self, code, day):
        #trade_columns = pd.read_csv('raw/GTA_SZL2_TRADE.csv')
        #self.trade = pd.read_csv(f'raw/SZL2_TRADE_{code}_{day[:6]}.csv', names=list(trade_columns))
        self.trade = pd.read_csv(f"data/{code}/{day}/trade.csv")

        self.trade.TradingTime = pd.to_datetime(self.trade.TradingTime)

        #self.trade = self.trade[self.trade.TradingDate==int(day)]
        #self.trade = self.trade[self.trade.TradeType=="F"]
        #print("loaded trades ", len(self.trade))
        #print("all trading time ", set(self.trade.TradingTime))
        self.trade = self.trade[self.trade['TradingTime'].dt.time.between(time(9, 30), time(14, 57))]
        #print("loaded trades ", len(self.trade))
        #self.trade = self.trade[(f'{self.day} 09:30:00'<self.trade.TradingTime)&(self.trade.TradingTime<f'{self.day} 14:57:00')]

        #print("order book index ", self.orderbook.index)
        self.is_trade = pd.DataFrame(index=self.orderbook.index,columns=['is_trade'])
        self.is_trade['is_trade'] = 0
        #print("all trading time ", set(self.trade.TradingTime))
        self.is_trade.loc[set(self.trade.TradingTime)] = 1
        #print("length of is_trade ", len(self.is_trade))
        #print(self.is_trade.value_counts())

    '''
        Common function
    '''

    def reset_seq(self, timesteps_per_episode=None, episode_idx=None):
        self.episode_idx = episode_idx
        #print('timesteps_per_episode ', timesteps_per_episode)
        if timesteps_per_episode == None:
            self.episode_start = 0
            self.episode_end = len(self.orderbook)
            self.episode_state = self.orderbook
        else:
            self.episode_start = timesteps_per_episode * episode_idx
            self.episode_end = min(self.episode_start + timesteps_per_episode, len(self.orderbook))
            self.episode_state = self.orderbook.iloc[self.episode_start:self.episode_end]

        self.episode_length = len(self.episode_state)

        #print("self.episode_start, self.episode_end ", self.episode_start, self.episode_end)
        episode_is_trade = self.is_trade.iloc[self.episode_start:self.episode_end]
        #print("self.is_trade ", self.is_trade)
        #print("episode_is_trade ", episode_is_trade)
        has_trade_index = np.where(episode_is_trade==1)[0]
        #ignore first T=50
        has_trade_index = has_trade_index[has_trade_index>self.T]
        #print("has_trade_index ", len(has_trade_index))
        #print("has_trade_index ", has_trade_index)
        self.index_iterator = iter(has_trade_index)


        self.cash = self.value_ = self.value = self.initial_value
        self.holding_pnl_total = self.trading_pnl_total = 0
        self.inventory = 0
        self.volume = 0
        self.episode_reward = 0
        self.mid_price_ = None
        self.action_his = []
        self.reward_dampened_pnl = 0
        self.reward_trading_pnl = 0
        self.reward_inventory_punishment = 0
        self.reward_spread_punishment = 0

        # log for trade
        self.logger = self.price.iloc[self.episode_start:self.episode_end].copy()
        #print("self.logger length, ", len(self.logger))
        columns=['ask_price', 'bid_price', 'trade_price', 'trade_volume', 'value', 'volume', 'cash', 'inventory']
        for column in columns:
            self.logger[column] = np.nan

        self.i = next(self.index_iterator)
        self.i_ = next(self.index_iterator)
        state = self.get_state_at_t(self.i-self.latency)

        if self.log >= 1:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state

    def reset_random(self, timesteps_per_episode=2000):
        self.episode_start = np.random.randint(0, len(self.orderbook) - timesteps_per_episode)
        self.episode_end = min(self.episode_start + timesteps_per_episode, len(self.orderbook))
        self.episode_state = self.orderbook.iloc[self.episode_start:self.episode_end]

        self.episode_length = len(self.episode_state)

        episode_is_trade = self.is_trade.iloc[self.episode_start:self.episode_end]
        has_trade_index = np.where(episode_is_trade==1)[0]
        #print("has_trade_index ", len(has_trade_index))
        has_trade_index = has_trade_index[has_trade_index>self.T]
        self.index_iterator = iter(has_trade_index)

        self.cash = self.value_ = self.value = self.initial_value
        self.holding_pnl_total = self.trading_pnl_total = 0
        self.inventory = 0
        self.volume = 0
        self.episode_reward = 0
        self.mid_price_ = None
        self.action_his = []
        self.reward_dampened_pnl = 0
        self.reward_trading_pnl = 0
        self.reward_inventory_punishment = 0
        self.reward_spread_punishment = 0

        # log for trade
        self.logger = self.price.iloc[self.episode_start:self.episode_end].copy()
        columns=['ask_price', 'bid_price', 'trade_price', 'trade_volume', 'value', 'volume', 'cash', 'inventory']
        for column in columns:
            self.logger[column] = np.nan

        self.i = next(self.index_iterator)
        self.i_ = next(self.index_iterator)
        state = self.get_state_at_t(self.i-self.latency)

        if self.log:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state

    def execute(self, actions):
        self.action_his.append(actions)
        # t
        self.mid_price, self.ask1_price, self.bid1_price, self.lob_spread = self.get_price_info(self.i)
        if self.mid_price_ == None:
            self.mid_price_ = self.mid_price

        orders = self.action2order(actions)
        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            orders['ask_price']=0
        elif self.inventory > 10*TRADE_UNIT:
            orders['bid_price']=0

        trade_price, trade_volume = self.match(orders)

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        if self.i >= len(self.logger):
           print("index overflow, ", self.i, len(self.logger))
        else:
           self.logger.iloc[self.i, -8:] = [orders['ask_price'], orders['bid_price'], trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        # if trade_volume:
        #     print(self.i, 'ask1:', self.ask1_price, 'bid1:', self.bid1_price, 'buy' if trade_volume>0 else 'sell', 'at', trade_price)
        if self.log >= 1:
            self.pbar.update(self.i_ - self.i)

        self.i = self.i_
        # Termination conditions
        terminal = False
        try:
            self.i_ = next(self.index_iterator)
        except:
            terminal = True

        reward = self.get_reward(trade_price, trade_volume)
        self.mid_price_ = self.mid_price

        # close position
        if terminal:
            trade_price, trade_volume = self.close_position()
            reward += self.get_reward(trade_price, trade_volume)

        self.episode_reward += reward

        # log for result
        if terminal:
            self.post_experiment(False)

        state = self.get_state_at_t(self.i-self.latency)

        return state, terminal, reward

    def match(self, actions):
        trade_volume = 0
        trade_price = 0
        ask_price, ask_volume, bid_price, bid_volume = actions.values()

        # trade
        now_t = self.trade[self.trade.TradingTime==self.episode_state.index[self.i]]
        now_trading_price_max = now_t.TradePrice.max()
        now_trading_price_max_v = now_t[now_t.TradePrice==now_trading_price_max].TradeVolume.sum()
        now_trading_price_min = now_t.TradePrice.min()
        now_trading_price_min_v = now_t[now_t.TradePrice==now_trading_price_min].TradeVolume.sum()

        # t - 1
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-1)

        # sell order
        if ask_price and ask_volume:
            if ask_price <= t_1_b1_price:
                # market order
                trade_price, trade_volume = t_1_b1_price, ask_volume
                # print("market order sell at", trade_price)
            else:
                # limit order
                if now_trading_price_max > ask_price:
                    # all deal
                    trade_price, trade_volume = ask_price, ask_volume
                    # print("limit order sell at", trade_price)

                # we assume that our quotes rest at the back of the queue
                elif now_trading_price_max == ask_price:
                    # deal probability: traded volume/all volume in this level
                    lob_depth = self.episode_state.iloc[self.i].ask_quantity_1
                    transac_prob = now_trading_price_max_v/(now_trading_price_max_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = ask_price, ask_volume

        # buy order
        if bid_price and bid_volume:
            if bid_price >= t_1_a1_price:
                # market order
                trade_price, trade_volume = t_1_a1_price, bid_volume
                # print("market order buy at", trade_price)
            else:
                if now_trading_price_min < bid_price:
                    trade_price, trade_volume = bid_price, bid_volume
                    # print("limit order buy at", trade_price)

                # we assume that our quotes rest at the back of the queue
                elif now_trading_price_min == bid_price:
                    lob_depth = self.episode_state.iloc[self.i].bid_quantity_1
                    transac_prob = now_trading_price_min_v/(now_trading_price_min_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = bid_price, bid_volume

        return trade_price, trade_volume

    def close_position(self):
        # t - 1
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-1)

        # Market order
        if self.inventory < 0:
            # Buy
            trade_price, trade_volume = t_1_a1_price, -self.inventory
            self.volume += trade_volume
        elif self.inventory > 0:
            # Sell
            trade_price, trade_volume = t_1_b1_price, -self.inventory
        else:
            trade_price, trade_volume = 0, 0

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        self.logger.iloc[self.i, -6:] = [trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        return trade_price, trade_volume

    def update_agent(self, trade_price, trade_volume):
        self.inventory_ = self.inventory
        self.inventory += trade_volume
        self.cash -= trade_volume*trade_price
        self.value = self.get_value(self.mid_price)

        volume = max(0, trade_volume*trade_price) # only count for buy
        self.volume += volume

    def get_price_info(self, i):
        price = self.price[self.price.index==self.episode_state.index[i]].iloc[-1]
        #print(price)

        bid1_price = price.bid1_price.item()
        ask1_price = price.ask1_price.item()
        bid1_price, ask1_price = round(bid1_price,2), round(ask1_price,2)
        mid_price = (bid1_price+ask1_price)/2
        spread = ask1_price - bid1_price

        return mid_price, ask1_price, bid1_price, spread

    def get_value(self, price):
        return self.cash + self.inventory*price

    '''
        For evaluation and save trading log
    '''

    def post_experiment(self, save=False):
        logger_wo_exit_market = self.logger[(self.logger.ask_price != 0) & (self.logger.bid_price != 0)]
        self.episode_avg_spread = (logger_wo_exit_market.ask_price - logger_wo_exit_market.bid_price).mean()
        self.episode_avg_position = self.logger.inventory.mean()
        self.episode_avg_abs_position = self.logger.inventory.abs().mean()
        self.episode_profit_ratio = self.value/(self.volume+1e-7)
        self.pnl = self.value - self.initial_value
        self.nd_pnl = self.pnl/self.episode_avg_spread
        self.pnl_map = self.pnl/(self.episode_avg_abs_position+1e-7)

        if self.log >= 1:
            print(
                "PnL:", self.pnl,
                "Holding PnL", self.holding_pnl_total,
                "Trading PnL", self.trading_pnl_total,
                "ND-PnL:", self.nd_pnl,
                "PnL-MAP:", self.pnl_map,
                "Trading volume:", self.volume,
                "Profit ratio:", self.episode_profit_ratio,
                "Averaged position:",self.episode_avg_position,
                "Averaged Abs position:",self.episode_avg_abs_position,
                "Averaged spread:", self.episode_avg_spread,
                "Episodic reward:", self.episode_reward
                )
            self.pbar.close()

        if self.log >= 2:
            trade_log = self.logger[(self.logger.trade_volume > 0)|(self.logger.trade_volume < 0)]
            for i in range(len(trade_log)):
                item = trade_log.iloc[i]
                if item.trade_volume > 0:
                    print(item.name, 'BUY at', item.trade_price, 'inventory', item.inventory, 'value', item.value)
                elif item.trade_volume < 0:
                    print(item.name, 'SELL at', item.trade_price, 'inventory', item.inventory, 'value', item.value)

        if save:
            now_time = time_module.strftime('%Y_%m_%d_%H_%M_%S', time_module.localtime())
            log_file = f"./log/{self.exp_name}_{self.code}_{self.day}_{now_time}.csv"
            self.logger.to_csv(log_file)
            print("Trading log saved to", log_file)

    def get_final_result(self):
        return dict(
            pnl=self.pnl,
            nd_pnl=self.nd_pnl,
            pnl_map=self.pnl_map,
            profit_ratio=self.episode_profit_ratio,
            avg_position=self.episode_avg_position,
            avg_abs_position=self.episode_avg_abs_position,
            avg_spread=self.episode_avg_spread,
            volume=self.volume,
            episode_reward=self.episode_reward
        )

#env_feature
from datetime import timedelta


class EnvFeature(BaseEnv):
    """
        Use this class to calculate your factor
    """
    def __init__(
            self,
            **kwargs
        ):
        super().__init__(**kwargs)

    def _get_market_state(self,t):
        data_300s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=300))].midprice
        data_600s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=600))].midprice
        data_1800s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=1800))].midprice
        rv_300s = getRealizedVolatility(data_300s,resample='s')*1e4
        rv_600s = getRealizedVolatility(data_600s,resample='s')*1e4
        rv_1800s = getRealizedVolatility(data_1800s,resample='s')*1e4
        rsi_300s = getRelativeStrengthIndex(data_300s)
        rsi_600s = getRelativeStrengthIndex(data_600s)
        rsi_1800s = getRelativeStrengthIndex(data_1800s)
        return [rv_300s, rv_600s, rv_1800s, rsi_300s, rsi_600s, rsi_1800s]

    def _get_order_strength_index(self,t):
        #data_10s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=10))]
        #data_60s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=60))]
        #data_300s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=300))]

        #svi_10s, sni_10s, lvi_10s, lni_10s, wvi_10s, wni_10s = getOrderStrengthIndex(data_10s)
        #svi_60s, sni_60s, lvi_60s, lni_60s, wvi_60s, wni_60s = getOrderStrengthIndex(data_60s)
        #svi_300s, sni_300s, lvi_300s, lni_300s, wvi_300s, wni_300s = getOrderStrengthIndex(data_300s)
        return [0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                ]

        #return [
        #    svi_10s, sni_10s, lvi_10s, lni_10s, wvi_10s, wni_10s,
        #    svi_60s, sni_60s, lvi_60s, lni_60s, wvi_60s, wni_60s,
        #    svi_300s, sni_300s, lvi_300s, lni_300s, wvi_300s, wni_300s
        #]



#env_continuous
import numpy as np
import pandas as pd
import random
import math


class EnvContinuous(EnvFeature):
    """
    """
    def __init__(
            self,
            code='600519',
            day='20191101',
            latency=1,
            T=50,
            # ablation states
            wo_lob_state=False,
            wo_market_state=False,
            wo_agent_state=False,
            # ablation rewards
            wo_dampened_pnl=False,
            wo_matched_pnl=False,
            wo_inv_punish=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.name = "Continuous"
        print("Environment:", self.name)
        self.code = code
        self.day = day2date(day)

        self.latency = latency
        self.T = T

        # ablation
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_agent_state = wo_agent_state
        self.r_da = 0 if wo_dampened_pnl else 1
        self.r_ma = 0 if wo_matched_pnl else 1
        self.r_ip = 0 if wo_inv_punish else 1

        # Inventory punishment factor
        self.theta = 0.01
        self.eta = 0.9

        self.init_states()

        self.load_orderbook(code=code, day=day)
        self.load_price(code=code, day=day)
        self.load_trade(code=code, day=day)
        #self.load_msg(code=code, day=day)

    def init_states(self):
        self.__states_space__ = dict()
        if not self.wo_lob_state:
            self.__states_space__['lob_state'] = dict(
                type='float',
                shape=(self.T,40,1)
                )
        if not self.wo_market_state:
            self.__states_space__['market_state'] = dict(
                type='float',
                shape=(24,)
                )
        if not self.wo_agent_state:
            self.__states_space__['agent_state'] = dict(
                type='float',
                shape=(24,)
                )

    def states(self):
        return self.__states_space__

    def actions(self):
        return dict(
                    type='float',
                    shape=(2,),
                    min_value=-1,
                    max_value=1
                )

    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        # action 1
        # actions in [0, 1]
        delta_price = actions[0]*0.05
        spread = actions[1]*0.1
        if self.inventory > 0:
            reservation = t_1_mid_price - delta_price
        elif self.inventory < 0:
            reservation = t_1_mid_price + delta_price
        else:
            reservation = t_1_mid_price
        ask_price = reservation + spread/2
        bid_price = reservation - spread/2

        # action 2
        # actions in [-1, 1]
        # delta_price = actions[0]*0.05
        # spread = abs(actions[1])*0.1
        # reservation = t_1_mid_price - delta_price
        # ask_price = reservation + spread/2
        # bid_price = reservation - spread/2

        # action 3
        # actions in [0, 1]
        # ask_price = t_1_a1_price + actions[0]*0.1
        # bid_price = t_1_b1_price - actions[1]*0.1
        # reservation = (ask_price + bid_price)/2
        # spread = ask_price - bid_price

        ask_price, bid_price = price_legal_check(ask_price, bid_price)

        # save for log
        self.reservation = reservation
        self.spread = spread

        orders = {
            'ask_price': ask_price,
            'ask_vol': -TRADE_UNIT,
            'bid_price': bid_price,
            'bid_vol': TRADE_UNIT
        }
        return orders

    def get_reward(self, trade_price, trade_volume):
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        matched_pnl = (self.mid_price - trade_price) * trade_volume

        # delta_inventory = abs(self.inventory) - abs(self.inventory_)
        # delta_inventory = max(0, delta_inventory)
        # inventory_punishment = self.theta * (delta_inventory/TRADE_UNIT)

        inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2

        # spread punishment
        if self.inventory:
            spread_punishment = 0
        else:
            spread_punishment = 100*self.spread if self.spread > 0.02 else 0

        reward = pnl - spread_punishment#self.r_ma * matched_pnl + self.r_da * dampened_pnl - self.r_ip * inventory_punishment - spread_punishment

        self.value_ = self.value

        return reward

    def get_state_at_t(self, t):
        self.__state__ = dict()

        if not self.wo_lob_state:
            lob = self.episode_state.iloc[t-self.T:t]
            #print("lob ", lob)
            mid_price = (lob.ask_price_1 + lob.bid_price_1)/2
            lob_normed = lob_norm(lob, mid_price)
            self.__state__['lob_state'] = np.expand_dims(np.array(lob_normed), -1)

        if not self.wo_market_state:
            self.__state__['market_state'] = self._get_market_state(t) + self._get_order_strength_index(t)

        if not self.wo_agent_state:
            self.__state__['agent_state'] = [self.inventory/(10*TRADE_UNIT)]*12 + [t / self.episode_length]*12

        return self.__state__



#env_discret
import numpy as np
import pandas as pd
import random
import math
from datetime import timedelta




class EnvDiscrete(EnvFeature):
    """
    """
    def __init__(
            self,
            code='000001',
            day='20191101',
            data_norm=True,
            latency=1,
            T=50,
            # ablation states
            wo_lob_state=False,
            wo_market_state=False,
            wo_agent_state=False,
            # ablation rewards
            wo_dampened_pnl=False,
            wo_matched_pnl=False,
            wo_inv_punish=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        print("Environment: EnvDiscrete")
        self.code = code
        self.day = day2date(day)

        self.latency = latency
        self.T = T

        # ablation
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_agent_state = wo_agent_state
        self.r_da = 0 if wo_dampened_pnl else 1
        self.r_ma = 0 if wo_matched_pnl else 1
        self.r_ip = 0 if wo_inv_punish else 1

        # Inventory punishment factor
        self.theta = 0.01
        self.eta = 0.5

        self.init_states()

        self.load_orderbook(code=code, day=day)
        self.load_price(code=code, day=day)
        self.load_trade(code=code, day=day)
        #self.load_msg(code=code, day=day)

    def init_states(self):
        self.__states_space__ = dict()
        if not self.wo_lob_state:
            self.__states_space__['lob_state'] = dict(
                type='float',
                shape=(self.T,40,1)
                )
        if not self.wo_market_state:
            self.__states_space__['market_state'] = dict(
                type='float',
                shape=(24,)
                )
        if not self.wo_agent_state:
            self.__states_space__['agent_state'] = dict(
                type='float',
                shape=(24,)
                )

    def states(self):
        return self.__states_space__

    def actions(self):
        return dict(
                    type='int',
                    num_values=5
                )

    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        ask_price, bid_price = 0, 0
        ask_volume, bid_volume = -TRADE_UNIT,TRADE_UNIT

        if actions in range(7):
            # limit order
            if actions == 0:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price
            elif actions == 1:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.01
            elif actions == 2:
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price
            elif actions == 3:
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price-0.01
            elif actions == 4:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.02
            elif actions == 5:
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price
            elif actions == 6:
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price-0.02

        elif actions==7:
            # market order to clode position
            if self.inventory < 0:
                bid_price, bid_volume = np.inf, -self.inventory
            elif self.inventory > 0:
                ask_price, ask_volume = 0.01, -self.inventory
            else:
                trade_price, trade_volume = 0, 0

        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            ask_price=0
            ask_volume=0
        elif self.inventory > 10*TRADE_UNIT:
            bid_price=0
            bid_volume=0

        orders = {
            'ask_price': ask_price,
            'ask_vol': ask_volume,
            'bid_price': bid_price,
            'bid_vol': bid_volume
        }

        return orders

    def get_reward(self, trade_price, trade_volume):
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        matched_pnl = (self.mid_price - trade_price) * trade_volume

        delta_inventory = abs(self.inventory) - abs(self.inventory_)
        # delta_inventory = max(0, delta_inventory)

        inventory_punishment = self.theta * (delta_inventory/TRADE_UNIT)
        # inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2
        reward = pnl
        # reward = self.r_ma * matched_pnl + self.r_da * dampened_pnl - self.r_ip * inventory_punishment
        self.value_ = self.value

        return reward

    def get_state_at_t(self, t):
        self.__state__ = dict()

        if not self.wo_lob_state:
            lob = self.episode_state.iloc[t-self.T:t]
            mid_price = (lob.ask_price_1 + lob.bid_price_1)/2
            lob_normed = lob_norm(lob, mid_price)
            self.__state__['lob_state'] = np.expand_dims(np.array(lob_normed), -1)

        if not self.wo_market_state:
            self.__state__['market_state'] = self._get_market_state(t) + self._get_order_strength_index(t)

        if not self.wo_agent_state:
            self.__state__['agent_state'] = [self.inventory/(10*TRADE_UNIT)]*12 + [t / self.episode_length]*12

        return self.__state__


#agent
from tensorforce.agents import Agent

def get_dueling_dqn_agent(
                        network,
                        environment=None,
                        states=None,
                        actions=None,
                        max_episode_timesteps=None,
                        batch_size=32,
                        learning_rate=1e-4,
                        horizon=1,
                        discount=0.99,
                        memory=200000,
                        device='gpu'
                        ):
    if environment != None:
        agent = Agent.create(
        agent='dueling_dqn',
        environment=environment,
        max_episode_timesteps=max_episode_timesteps,
        network=network,
        config=dict(device=device),
        memory=memory,
        batch_size=batch_size,
        learning_rate=learning_rate,
        horizon=horizon,
        discount=discount,
        parallel_interactions=10,
    )
    else:
        agent = Agent.create(
            agent='dueling_dqn',
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            memory=memory,
            batch_size=batch_size,
            learning_rate=learning_rate,
            horizon=horizon,
            discount=discount,
            parallel_interactions=10,
        )
    return agent

def get_ppo_agent(
                network,
                environment=None,
                states=None,
                actions=None,
                max_episode_timesteps=None,
                batch_size=32,
                learning_rate=1e-3,
                horizon=None,
                discount=0.99,
                device='gpu'
                ):
    if environment != None:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            batch_size=batch_size,
            learning_rate=learning_rate,
            discount=discount,
            parallel_interactions=10,
        )
    else:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            batch_size=batch_size,
            learning_rate=learning_rate,
            discount=discount,
            parallel_interactions=10,
        )

    return agent



#network
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from tensorflow import keras
from keras import backend as K

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True                                #按需分配显存
K.set_session(tf.compat.v1.Session(config=config))

def get_lob_model(latent_dim, T):
    lob_state = keras.layers.Input(shape=(T, 40, 1))

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(lob_state)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 5), strides=(1, 5))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 4))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    attn_input = conv_reshape
    attn_input_last = attn_input[:,-1:,:]

    multi_head_attn_layer_1 = keras.layers.MultiHeadAttention(num_heads=10, key_dim=16, output_shape=64)

    attn_output, weight = multi_head_attn_layer_1(attn_input_last, attn_input, return_attention_scores=True)

    attn_output = keras.layers.Flatten()(attn_output)

    # add Batch Normalization
    # attn_output = keras.layers.BatchNormalization()(attn_output)

    # add Layer Normalization
    # attn_output = keras.layers.LayerNormalization()(attn_output)

    return keras.models.Model(lob_state, attn_output)


def get_fclob_model(latent_dim,T):
    print("This is the FC-LOB model")
    lob_state = keras.layers.Input(shape=(T, 40, 1))

    dense_input = keras.layers.Flatten()(lob_state)

    dense_output = keras.layers.Dense(1024, activation='leaky_relu')(dense_input)
    dense_output = keras.layers.Dense(256, activation='leaky_relu')(dense_input)
    dense_output = keras.layers.Dense(latent_dim, activation='leaky_relu')(dense_input)

    return keras.models.Model(lob_state, dense_output)

def compute_output_shape(input_shape):
    return (input_shape[0], 64)

def get_pretrain_model(model, T):
    lob_state = keras.layers.Input(shape=(T, 40, 1))
    embedding = model(lob_state)
    output = keras.layers.Dense(3, activation='softmax')(embedding)

    return keras.models.Model(lob_state, output)

def get_model(lob_model, T, with_lob_state=True, with_market_state=True, with_agent_state=True):
    input_ls = list()
    dense_input = list()
    if with_lob_state:
        lob_state = keras.layers.Input(shape=(T, 40, 1))
        encoder_outputs = lob_model(lob_state)
        input_ls.append(lob_state)
        dense_input.append(encoder_outputs)
    else:
        print('w/o lob state!')

    if with_agent_state:
        agent_state = keras.layers.Input(shape=(24,))
        input_ls.append(agent_state)
        dense_input.append(agent_state)
    else:
         print('w/o agent state!')

    if with_market_state:
        market_state = keras.layers.Input(shape=(24,))
        input_ls.append(market_state)
        dense_input.append(agent_state)
    else:
        print('w/o market state!')

    dense_input = keras.layers.concatenate(dense_input, axis=1)

    dense_output = keras.layers.Dense(64, activation='leaky_relu')(dense_input)

    return keras.models.Model(input_ls, dense_output)

#if __name__ == '__main__':
#    get_lob_model(64,50).summary()

#main
import os
import argparse
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import pyrallis
from dataclasses import asdict, dataclass

from tensorforce.environments import Environment



@dataclass
class TrainConfig:
    # Experiment
    code: str = '600519'
    device: str = "cpu"
    latency: int = 0
    time_window: int = 50
    log: int = 0
    exp_name: str = ''
    # Agent
    agent_type: str = 'ppo' # ppo/dueling dqn
    learning_rate: int = 1e-4
    horizon: int = 1
    env_type: str = 'continuous' # continuous/discrete
    load: bool = False
    agent_load_dir: str = 'agent'
    save: bool = False,
    agent_save_dir: str = 'agent'
    # Ablation
    wo_pretrain: bool = True
    wo_attnlob: bool = False
    wo_lob_state: bool = False
    wo_market_state: bool = False
    wo_dampened_pnl: bool = False
    wo_matched_pnl: bool = False
    wo_inv_punish: bool = False


def init_env(day, config):
    if config['env_type'] == 'continuous':
        env = EnvContinuous
    elif config['env_type'] == 'discrete':
        env = EnvDiscrete

    environment = env(
        code=config['code'],
        day=day,
        latency=config['latency'],
        T=config['time_window'],
        # state ablation
        wo_lob_state=config['wo_lob_state'],
        wo_market_state=config['wo_market_state'],
        # reward ablation
        wo_dampened_pnl=config['wo_dampened_pnl'],
        wo_matched_pnl=config['wo_matched_pnl'],
        wo_inv_punish=config['wo_inv_punish'],
        # exp setting
        experiment_name=config['exp_name'],
        log=config['log'],
        )
    return environment

def init_agent(environment, config):
    kwargs=dict()
    if config['agent_type'] == 'dueling_dqn':
        get_agent = get_dueling_dqn_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']
    elif config['agent_type'] == 'ppo':
        get_agent = get_ppo_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']

    if config['wo_pretrain']:
        print("Ablation: pretrain")
        lob_model = get_lob_model(64,config['time_window'])
        lob_model.compute_output_shape = compute_output_shape
    else:
        pretrain_model_dir = f'./ckpt/pretrain_model_' + config['code']
        model = get_lob_model(64,config['time_window'])
        model.compute_output_shape = compute_output_shape
        model_pretrain = get_pretrain_model(model,config['time_window'])
        checkpoint_filepath = pretrain_model_dir + '/weights'
        model_pretrain.load_weights(checkpoint_filepath)
        lob_model = model_pretrain.layers[1]

    if config['wo_attnlob']:
        print("Ablation: attnlob")
        lob_model = get_fclob_model(64,config['time_window'])

    model = get_model(
        lob_model,
        config['time_window'],
        with_lob_state= not config['wo_lob_state'],
        with_market_state= not config['wo_market_state']
        )
    agent = get_agent(model, environment=environment, max_episode_timesteps=1000, device=config['device'], **kwargs)

    if config['load']:
        model = keras.models.load_model(keras_model_dir)
        model.layers[1].compute_output_shape = compute_output_shape
        agent = get_agent(model, environment=environment, max_episode_timesteps=1000, device=config['device'], **kwargs)
        agent.restore(config['agent_load_dir'], filename='cppo', format='numpy')

    return agent

def train_a_day(environment, agent, train_result):
    num_episodes = len(environment.orderbook)//num_step_per_episode
    data_collector = list()
    for idx in tqdm(range(num_episodes)):
        episode_states = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        terminal = False
        while not terminal:
            episode_states.append(states)
            actions = agent.act(states=states, independent=True)
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        data_collector.append([episode_states, episode_actions, episode_terminal, episode_reward])

        agent.experience(
            states=episode_states,
            actions=episode_actions,
            terminal=episode_terminal,
            reward=episode_reward
        )

        agent.update()

        save_episode_result(environment, train_result)

    return episode_states, episode_actions, episode_reward

def test_a_day(environment, agent, test_result):
    num_episodes = len(environment.orderbook)//num_step_per_episode
    for idx in tqdm(range(num_episodes)):

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        terminal = False
        while not terminal:
            actions = agent.act(
                states=states, independent=True
            )
            states, terminal, reward = environment.execute(actions=actions)

        save_episode_result(environment, test_result)

def train(agent, train_result, config):
    for day in train_days:
        environment = init_env(day, config)
        train_a_day(environment, agent, train_result)

def test(agent, test_result, config):
    for day in test_days:
        environment = init_env(day, config)
        test_a_day(environment, agent, test_result)

def save_episode_result(environment, test_result):
    res_dict = environment.get_final_result()
    date = environment.day
    idx = environment.episode_idx

    test_result.loc[date+'_'+str(idx)] = [res_dict['pnl'], res_dict['nd_pnl'], res_dict['avg_abs_position'], res_dict['profit_ratio'], res_dict['volume']]

def gather_test_results(test_result):
    day_list = list(test_result.index)
    for i in range(len(day_list)):
        day_list[i] = day_list[i][:10]
    day_list = set(day_list)
    gathered_results = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    for day in day_list:
        result = test_result[test_result.index.str.contains(day)]
        pnl = result.PnL.sum()
        nd_pnl = result['ND-PnL'].sum()
        ap = result.average_position.mean()
        volume = (result.PnL/result.profit_ratio).sum()
        pr = pnl/volume
        gathered_results.loc[day] = [pnl,nd_pnl,ap,pr,volume]
    gathered_results=gathered_results.sort_index()
    return gathered_results

def save_agent(agent, config):
    # save agent network
    agent.model.policy.network.keras_model.save(keras_model_dir)
    # Save agent
    agent.save(config['agent_save_dir'], filename=agent, format='numpy')

#@pyrallis.wrap()
def main(config: TrainConfig):
    config = asdict(config)

    environment = init_env(train_days[0], config)
    agent = init_agent(environment, config)

    train_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    for _ in range(n_train_loop):
        train(agent, train_result, config)
        if config['save']:
            save_agent(agent, config)

    test_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    test(agent, test_result, config)
    daily_test_results = gather_test_results(test_result)

keras_model_dir='model'
##train_days=['20230320', '20230321']
train_days=['20230320']
test_days=['20230322']
num_step_per_episode = 2000
n_train_loop = 1
#
config = TrainConfig()
main(config)
