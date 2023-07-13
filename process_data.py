import os
os.chdir('c:\work\china-data\sh-mix')
os.listdir(".")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_raw = pd.read_csv("20230320\Snapshot.small.csv")
colunm_names = ['SecurityID','DateTime','PreClosePx','OpenPx','HighPx','LowPx','LastPx','TotalVolumeTrade','TotalValueTrade','InstrumentStatus','BidPrice1','BidPrice2','BidPrice3','BidPrice4','BidPrice5','BidPrice6','BidPrice7','BidPrice8','BidPrice9','BidPrice10','BidOrderQty1','BidOrderQty2','BidOrderQty3','BidOrderQty4','BidOrderQty5','BidOrderQty6','BidOrderQty7','BidOrderQty8','BidOrderQty9','BidOrderQty10','BidNumOrders1','BidNumOrders2','BidNumOrders3','BidNumOrders4','BidNumOrders5','BidNumOrders6','BidNumOrders7','BidNumOrders8','BidNumOrders9','BidNumOrders10','BidOrders1','BidOrders2','BidOrders3','BidOrders4','BidOrders5','BidOrders6','BidOrders7','BidOrders8','BidOrders9','BidOrders10','BidOrders11','BidOrders12','BidOrders13','BidOrders14','BidOrders15','BidOrders16','BidOrders17','BidOrders18','BidOrders19','BidOrders20','BidOrders21','BidOrders22','BidOrders23','BidOrders24','BidOrders25','BidOrders26','BidOrders27','BidOrders28','BidOrders29','BidOrders30','BidOrders31','BidOrders32','BidOrders33','BidOrders34','BidOrders35','BidOrders36','BidOrders37','BidOrders38','BidOrders39','BidOrders40','BidOrders41','BidOrders42','BidOrders43','BidOrders44','BidOrders45','BidOrders46','BidOrders47','BidOrders48','BidOrders49','BidOrders50','OfferPrice1','OfferPrice2','OfferPrice3','OfferPrice4','OfferPrice5','OfferPrice6','OfferPrice7','OfferPrice8','OfferPrice9','OfferPrice10','OfferOredrQty1','OfferOredrQty2','OfferOredrQty3','OfferOredrQty4','OfferOredrQty5','OfferOredrQty6','OfferOredrQty7','OfferOredrQty8','OfferOredrQty9','OfferOredrQty10','OfferNumOrders1','OfferNumOrders2','OfferNumOrders3','OfferNumOrders4','OfferNumOrders5','OfferNumOrders6','OfferNumOrders7','OfferNumOrders8','OfferNumOrders9','OfferNumOrders10','OfferOrders1','OfferOrders2','OfferOrders3','OfferOrders4','OfferOrders5','OfferOrders6','OfferOrders7','OfferOrders8','OfferOrders9','OfferOrders10','OfferOrders11','OfferOrders12','OfferOrders13','OfferOrders14','OfferOrders15','OfferOrders16','OfferOrders17','OfferOrders18','OfferOrders19','OfferOrders20','OfferOrders21','OfferOrders22','OfferOrders23','OfferOrders24','OfferOrders25','OfferOrders26','OfferOrders27','OfferOrders28','OfferOrders29','OfferOrders30','OfferOrders31','OfferOrders32','OfferOrders33','OfferOrders34','OfferOrders35','OfferOrders36','OfferOrders37','OfferOrders38','OfferOrders39','OfferOrders40','OfferOrders41','OfferOrders42','OfferOrders43','OfferOrders44','OfferOrders45','OfferOrders46','OfferOrders47','OfferOrders48','OfferOrders49','OfferOrders50','NumTrades','IOPV','TotalBidQty','TotalOfferQty','WeightedAvgBidPx','WeightedAvgOfferPx','TotalBidNumber','TotalOfferNumber','BidTradeMaxDuration','OfferTradeMaxDuration','NumBidOrders','NumOfferOrders','WithdrawBuyNumber','WithdrawBuyAmount','WithdrawBuyMoney','WithdrawSellNumber','WithdrawSellAmount','WithdrawSellMoney','ETFBuyNumber','ETFBuyAmount','ETFBuyMoney','ETFSellNumber','ETFSellAmount','ETFSellMoney','AvgPx','ClosePx','MsgSeqNum','SendingTime','WarLowerPx','TradingPhaseCode','NumImageStatus']
df_raw.columns = colunm_names

df=df_raw[['SecurityID','DateTime']
+['BidPrice1','BidPrice2','BidPrice3','BidPrice4','BidPrice5','BidPrice6','BidPrice7','BidPrice8','BidPrice9','BidPrice10','BidOrderQty1','BidOrderQty2','BidOrderQty3','BidOrderQty4','BidOrderQty5','BidOrderQty6','BidOrderQty7','BidOrderQty8','BidOrderQty9','BidOrderQty10']
+['OfferPrice1','OfferPrice2','OfferPrice3','OfferPrice4','OfferPrice5','OfferPrice6','OfferPrice7','OfferPrice8','OfferPrice9','OfferPrice10','OfferOredrQty1','OfferOredrQty2','OfferOredrQty3','OfferOredrQty4','OfferOredrQty5','OfferOredrQty6','OfferOredrQty7','OfferOredrQty8','OfferOredrQty9','OfferOredrQty10']
]

df.loc[:, 'Mid'] = 0.5 * (df['OfferPrice1'] + df['BidPrice1'])
def rolling_mean(a, n=11) :
    padded_column = np.pad(a, (n-1, 0), mode='constant', constant_values=np.nan)
    return pd.Series([np.mean(padded_column[i:i+n]) for i in range(len(a))], index=a.index).shift(-n+1)

df = df.sort_values(by=['SecurityID','DateTime']).reset_index(level=[0], drop=True).sort_index()

df['MidMean'] = (
    df
    .groupby('SecurityID')['Mid']
    .apply(rolling_mean)
)

df.dropna(inplace=True)
df = df[df['BidPrice1'] != 0]
df = df[df['OfferPrice1'] != 0]
df['MidReturn'] = (df['MidMean'] - df['Mid']) / df['Mid']


df['Label'] = pd.cut(x=df['MidReturn'], bins=[-np.inf, -0.00002, 0.00002, np.inf], labels=[1, 2, 3])

bid_price_cols = [f"BidPrice{index}" for index in range(1, 11)]
ask_price_cols = [f"OfferPrice{index}" for index in range(1, 11)]
bid_qty_cols = [f"BidOrderQty{index}" for index in range(1, 11)]
ask_qty_cols = [f"OfferOredrQty{index}" for index in range(1, 11)]
all_columns = bid_price_cols + ask_price_cols + bid_qty_cols + ask_qty_cols


df[all_columns] = df.groupby('SecurityID')[all_columns].transform(lambda x: (x - x.mean()) / x.std())


df[df['SecurityID']==600519].plot(x='DateTime', y=['BidPrice1', 'OfferPrice1'])
plt.show()

df['Label'].value_counts()


# plot quantile graph of value1
fig, ax = plt.subplots()
ax.set_title('Quantile plot of value1')
ax.set_ylabel('BidPrice1')
df['BidPrice1'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).plot(kind='line', ax=ax)

plt.show()