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

df11 = pd.DataFrame()
df22 = pd.DataFrame()

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
df11['pnl1'] = data

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
df22['pnl2'] = data


df_merged = pd.concat([df1['pnl1'], df2['pnl2']], ignore_index=True, axis=1)
df_merged.columns = ['pnl with features', 'pnl without features']
df_merged = df_merged[(df_merged['pnl with features'].notnull())]
df_merged.plot(y=['pnl with features', 'pnl without features'])
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL')
plt.title('PnL with/without market features')
plt.legend()
plt.show()

df_merged2 = pd.concat([df11['pnl1'], df22['pnl2']], ignore_index=True, axis=1)
df_merged2.columns = ['pnl1', 'pnl2']
df_merged2 = df_merged2[(df_merged2['pnl1'].notnull())]

df_merged2.describe()

df_merged2['return1'] = df_merged2['pnl1'].pct_change()
df_merged2['return2'] = df_merged2['pnl2'].pct_change()

mean_return1 = df_merged2['pnl1'].mean()
mean_return2 = df_merged2['pnl2'].mean()
std_dev1 = df_merged2['pnl1'].std()
std_dev2 = df_merged2['pnl2'].std()
sharpe_ratio1 = mean_return1 / std_dev1 * np.sqrt(252)
sharpe_ratio2 = mean_return2 / std_dev2 * np.sqrt(252)
max_drawdown1 = (df_merged2['pnl1'].cummax() - df_merged2['pnl1']) / df_merged2['pnl1'].cummax()
max_drawdown2 = (df_merged2['pnl2'].cummax() - df_merged2['pnl2']) / df_merged2['pnl2'].cummax()


# Print performance statistics
print("Mean: {:.2} {:.2}".format(mean_return1, mean_return2))
print("Daily standard deviation: {:.2}".format(std_dev1))
print("Sharpe ratio: {:.2f}".format(sharpe_ratio1))
print("Max drawdown: {:.2%}".format(max_drawdown1))

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



import pandas as pd
import numpy as np
from tabulate import tabulate

# Load PnL data from csv files into pandas dataframes
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df1['pnl'] = df_merged2['pnl1']
df1.dropna(inplace=True)
df2['pnl'] = df_merged2['pnl2']
df2.dropna(inplace=True)
# Compute daily returns
df1['return'] = df1['pnl'].pct_change()
df2['return'] = df2['pnl'].pct_change()
df1 = df1.replace([np.inf, -np.inf], np.nan).dropna(subset=['return'])
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=['return'])

# Compute performance statistics
stats1 = {
    'Mean': df1['return'].mean(),
    'Std Dev': df1['return'].std(),
    'Sharpe Ratio': df1['return'].mean() / df1['return'].std() * np.sqrt(252),
    'Max Drawdown': (df1['pnl'].cummax() - df1['pnl']) / df1['pnl'].cummax(),
    'VaR (95%)': df1['return'].quantile(0.05),
    'CVaR (95%)': df1['return'][df1['return'] <= df1['return'].quantile(0.05)].mean(),
    'Sortino Ratio': (df1['return'].mean() - 0.02) / df1['return'][df1['return'] < 0].std() * np.sqrt(252)
}

stats2 = {
    'Mean': df2['return'].mean(),
    'Std Dev': df2['return'].std(),
    'Sharpe Ratio': df2['return'].mean() / df2['return'].std() * np.sqrt(252),
    'Max Drawdown': (df2['pnl'].cummax() - df2['pnl']) / df2['pnl'].cummax(),
    'VaR (95%)': df2['return'].quantile(0.05),
    'CVaR (95%)': df2['return'][df2['return'] <= df2['return'].quantile(0.05)].mean(),
    'Sortino Ratio': (df2['return'].mean() - 0.02) / df2['return'][df2['return'] < 0].std() * np.sqrt(252)
}

# Format statistics as a table using tabulate
table = tabulate([stats1, stats2], headers=['Statistic', 'Series 1', 'Series 2'], tablefmt='pipe')

# Print the table
print(table)



data = []
with open('discret.1ms.2day.1loop.log', 'r') as f:
    for line in f:
        field = float(line.split()[8])
        if

for line in lines:
    try:
        field = float(line.split()[8])
        data.append(field)
    except IndexError:
        print(f"Warning: line '{line.strip()}' has fewer than 7 fields")



import pandas as pd


import pandas as pd

def plot_one_log(log_file_name, title):
    # Open the file for reading with the "utf-8" encoding
    with open(log_file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Define the patterns to extract values from
    patterns = ['PnL:', 'Holding PnL', 'Trading PnL', 'ND-PnL(pnl/spread):', 'PnL-MAP(pnl/pos):',
                'Trading volume:', 'Profit ratio:', 'Averaged position:',
                'Averaged Abs position:', 'Averaged spread:', 'Episodic reward:']

    # Create a dictionary to store the extracted values
    data = {pattern: [] for pattern in patterns}

    # Loop through the lines in the file
    for line in lines:
        # Loop through the patterns and check if the line matches each one
        for pattern in patterns:
            if pattern in line:
                # Try to extract the value and add it to the appropriate list in the dictionary
                try:
                    value = float(line.split(pattern + ' ')[1].split(' ')[0])
                    data[pattern].append(value)
                except (IndexError, ValueError):
                    # If the line doesn't have a value following the pattern, or the value is not a float, skip it
                    pass

    # Ensure all the lists in the data dictionary have the same length
    max_length = max(len(v) for v in data.values())
    for k in data.keys():
        data[k] += [None] * (max_length - len(data[k]))

    # Create a dataframe from the extracted values
    df = pd.DataFrame(data)
    df.columns = df.columns.str.rstrip(':')
    # Print the dataframe
    #print(df)

    df['CumPnL'] = df['PnL'].cumsum()
    df['CumTradingPnl'] = df['Trading PnL'].cumsum()
    df['CumHoldingPnl'] = df['Holding PnL'].cumsum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.index, df[['CumPnL', 'CumTradingPnl', 'CumHoldingPnl']])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PnL(CNY)')
    ax.set_title(title)
    fig.tight_layout()
    ax.xaxis.labelpad = 15

    plt.show()
plot_one_log('discret.1ms.2day.1loop.log', 'Discrete')
plot_one_log('continuous.1ms.2day.1loop.log', 'Continuous')