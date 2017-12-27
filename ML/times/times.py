import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from convert_data_to_timeseries import convert_data_to_timeseries

input_file = 'data_timeseries.txt'


'''
column_num = 2
data_timeseries = convert_data_to_timeseries(input_file, column_num)

start = '2008'
end = '2015'

plt.figure()
data_timeseries[start:end].plot()
plt.title('Data from {} to {}'.format(start, end))

start = '2007-2'
end = '2007-11'

plt.figure()
data_timeseries[start:end].plot()
plt.title('Data from {} to {}'.format(start, end))

plt.show()
'''
data1 = convert_data_to_timeseries(input_file,2)
data2 = convert_data_to_timeseries(input_file,3)

dataframe = pd.DataFrame({'first':data1, 'second':data2})
print('\nMaximum:\n', dataframe.max())
print('\nMinimum:\n', dataframe.min())
print('\nMean:\n', dataframe.mean())
print('\nMean row-wise:\n', dataframe.mean(1)[:10])


dataframe['1952':'1955'].plot()
plt.title('Data overlapped on top of each other')

plt.figure()
difference = dataframe['1952':'1955']['first'] - dataframe['1952':'1955']['second']
difference.plot()
plt.title('Difference (first - second)')

dataframe[(dataframe['first'] > 60) & (dataframe['second'] < 20)].plot()
plt.title('first > 60 & second < 20')


pd.rolling_mean(dataframe, window=24).plot()
plt.title('rolling_mean window = 24')
print('\nCorrelation coefficients:\n', dataframe.corr())

plt.figure()
pd.rolling_corr(dataframe['first'], dataframe['second'], window=60).plot()
plt.title('rolling_corr window = 60')
plt.show()