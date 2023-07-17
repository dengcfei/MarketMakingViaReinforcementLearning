import matplotlib.pyplot as plt
import os
from datetime import datetime, time
import pandas as pd
os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning')
os.listdir(".")

lines = []
with open('pnl.7.log', 'r') as f:
    for line in f:
        lines.append(line)
data = []
for line in lines:
    try:
        field = float(line.split()[7])
        data.append(field)
    except IndexError:
        print(f"Warning: line '{line.strip()}' has fewer than 7 fields")

plt.plot(data)
plt.show()
import numpy as np


import pandas as pd
os.chdir(r'c:\work\china-data\MarketMakingViaReinforcementLearning')
os.listdir(".")

df1 = pd.DataFrame()
df2 = pd.DataFrame()

lines = []
with open('pnl.8.detail.log', 'r') as f:
    for line in f:
        lines.append(line)
data = []
for line in lines:
    try:
        field = float(line.split()[8])
        data.append(field)
    except IndexError:
        print(f"Warning: line '{line.strip()}' has fewer than 7 fields")
cumulative_sum = np.cumsum(data)
plt.plot(cumulative_sum)
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL')
plt.title('PnL without latency')
plt.legend()
plt.show()
df1['pnl1'] = cumulative_sum

lines = []
with open('pnl.10.detail.log', 'r') as f:
    for line in f:
        lines.append(line)
data = []
for line in lines:
    try:
        field = float(line.split()[8])
        data.append(field)
    except IndexError:
        print(f"Warning: line '{line.strip()}' has fewer than 7 fields")
cumulative_sum = np.cumsum(data)
plt.plot(cumulative_sum)
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL')
plt.title('PnL without market features')
plt.legend()
plt.show()
df2['pnl2'] = cumulative_sum


df_merged = pd.concat([df1['pnl1'], df2['pnl2']], ignore_index=True, axis=1)
df_merged.columns = ['pnl with features', 'pnl without features']
df_merged = df_merged[(df_merged['pnl with features'].notnull())]
df_merged.plot(y=['pnl with features', 'pnl without features'])
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL')
plt.title('PnL with/without market features')
plt.legend()
plt.show()

pd.Series(data).describe()



lines = []
with open('pnl.9.detail.log', 'r') as f:
    for line in f:
        lines.append(line)
data = []
for line in lines:
    try:
        field = float(line.split()[8])
        data.append(field)
    except IndexError:
        print(f"Warning: line '{line.strip()}' has fewer than 7 fields")
cumulative_sum = np.cumsum(data)
plt.plot(cumulative_sum)
plt.show()