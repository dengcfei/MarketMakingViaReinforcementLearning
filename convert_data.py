import os
#os.chdir('c:\work\china-data\MarketMakingViaReinforcementLearning')
#os.listdir(".")
#
import pandas as pd

date_str='20230322'

def process_one(date_str):

    df = pd.read_csv(f'data/600519/{date_str}/Snapshot.csv')
    df.columns = ['SecurityID','DateTime','PreClosePx','OpenPx','HighPx','LowPx','LastPx','TotalVolumeTrade','TotalValueTrade','InstrumentStatus','BidPrice1','BidPrice2','BidPrice3','BidPrice4','BidPrice5','BidPrice6','BidPrice7','BidPrice8','BidPrice9','BidPrice10','BidOrderQty1','BidOrderQty2','BidOrderQty3','BidOrderQty4','BidOrderQty5','BidOrderQty6','BidOrderQty7','BidOrderQty8','BidOrderQty9','BidOrderQty10','BidNumOrders1','BidNumOrders2','BidNumOrders3','BidNumOrders4','BidNumOrders5','BidNumOrders6','BidNumOrders7','BidNumOrders8','BidNumOrders9','BidNumOrders10','BidOrders1','BidOrders2','BidOrders3','BidOrders4','BidOrders5','BidOrders6','BidOrders7','BidOrders8','BidOrders9','BidOrders10','BidOrders11','BidOrders12','BidOrders13','BidOrders14','BidOrders15','BidOrders16','BidOrders17','BidOrders18','BidOrders19','BidOrders20','BidOrders21','BidOrders22','BidOrders23','BidOrders24','BidOrders25','BidOrders26','BidOrders27','BidOrders28','BidOrders29','BidOrders30','BidOrders31','BidOrders32','BidOrders33','BidOrders34','BidOrders35','BidOrders36','BidOrders37','BidOrders38','BidOrders39','BidOrders40','BidOrders41','BidOrders42','BidOrders43','BidOrders44','BidOrders45','BidOrders46','BidOrders47','BidOrders48','BidOrders49','BidOrders50','OfferPrice1','OfferPrice2','OfferPrice3','OfferPrice4','OfferPrice5','OfferPrice6','OfferPrice7','OfferPrice8','OfferPrice9','OfferPrice10','OfferOredrQty1','OfferOredrQty2','OfferOredrQty3','OfferOredrQty4','OfferOredrQty5','OfferOredrQty6','OfferOredrQty7','OfferOredrQty8','OfferOredrQty9','OfferOredrQty10','OfferNumOrders1','OfferNumOrders2','OfferNumOrders3','OfferNumOrders4','OfferNumOrders5','OfferNumOrders6','OfferNumOrders7','OfferNumOrders8','OfferNumOrders9','OfferNumOrders10','OfferOrders1','OfferOrders2','OfferOrders3','OfferOrders4','OfferOrders5','OfferOrders6','OfferOrders7','OfferOrders8','OfferOrders9','OfferOrders10','OfferOrders11','OfferOrders12','OfferOrders13','OfferOrders14','OfferOrders15','OfferOrders16','OfferOrders17','OfferOrders18','OfferOrders19','OfferOrders20','OfferOrders21','OfferOrders22','OfferOrders23','OfferOrders24','OfferOrders25','OfferOrders26','OfferOrders27','OfferOrders28','OfferOrders29','OfferOrders30','OfferOrders31','OfferOrders32','OfferOrders33','OfferOrders34','OfferOrders35','OfferOrders36','OfferOrders37','OfferOrders38','OfferOrders39','OfferOrders40','OfferOrders41','OfferOrders42','OfferOrders43','OfferOrders44','OfferOrders45','OfferOrders46','OfferOrders47','OfferOrders48','OfferOrders49','OfferOrders50','NumTrades','IOPV','TotalBidQty','TotalOfferQty','WeightedAvgBidPx','WeightedAvgOfferPx','TotalBidNumber','TotalOfferNumber','BidTradeMaxDuration','OfferTradeMaxDuration','NumBidOrders','NumOfferOrders','WithdrawBuyNumber','WithdrawBuyAmount','WithdrawBuyMoney','WithdrawSellNumber','WithdrawSellAmount','WithdrawSellMoney','ETFBuyNumber','ETFBuyAmount','ETFBuyMoney','ETFSellNumber','ETFSellAmount','ETFSellMoney','AvgPx','ClosePx','MsgSeqNum','SendingTime','WarLowerPx','TradingPhaseCode','NumImageStatus']

#df['DateTime'] = df['DateTime'] * 1000
#df['DateTime'] = df['DateTime']
#df_600519 = df[df['SecurityID'] == 600519]
#df_600519 = df_600519.rename(columns =
#                             {
#                                 'DateTime' : 'timestamp',
#                                 'BidPrice1' : 'bid1_price',
#                                 'OfferPrice1' : 'ask1_price',
#                             })
#df_600519['midprice'] = 0.5 * (df_600519['bid1_price'] + df_600519['ask1_price'])
#df_600519.to_csv(r'data/600519/{date_str}/price.csv', index=False)


    df_order = pd.read_csv(f'data/600519/{date_str}/Entrust.csv')
    df_order.columns = ['SecurityID','CommissionTime','CommissionOrderNo','CommissionPx','COmmissionAmount','TradeMark','OrderCategory','CommissionNo','ChannelNo','SerialNo']

    df_order_600519 = df_order[df_order['SecurityID'] == 600519]
    df_order_600519 = df_order_600519.rename(columns =
                                 {
                                     'CommissionTime' : 'TradingTime',
                                 })
    df_order_600519['TradingTime'] = pd.to_datetime(df_order_600519['TradingTime'], format='%Y%m%d%H%M%S%f')
    df_order_600519.to_csv(f'data/600519/{date_str}/order.csv', index=False)


    df_trade = pd.read_csv(f'data/600519/{date_str}/Tick.csv')
    df_trade.columns = ['SecurityID','TradeTime','TradePx','TradeAmount','TradeValue','OrderNo','OfferNo','TradeNo','ChannelCode','SBMark','SerialNo','SendingTime']

    df_trade_600519 = df_trade[df_trade['SecurityID'] == 600519]
    df_trade_600519 = df_trade_600519.rename(columns =
                                 {
                                     'TradeTime' : 'TradingTime',
                                     'TradePx' : 'TradePrice',
                                     'TradeAmount' : 'TradeVolume',
                                 })
    df_trade_600519['TradingTime'] = pd.to_datetime(df_trade_600519['TradingTime'], format='%Y%m%d%H%M%S%f')
    df_trade_600519.to_csv(f'data/600519/{date_str}/trade.csv', index=False)

for date_str in ['20230323', '20230324', '20230327', '20230328', '20230329', '20230330', '20230331']:
    process_one(date_str)
