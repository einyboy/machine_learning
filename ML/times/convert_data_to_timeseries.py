import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_data_to_timeseries(input_file, column, verbose=True):
    data = np.loadtxt(input_file, delimiter=',')
    start_date = str(int(data[0,0])) + '-' + str(int(data[0,1]))
    end_date = str(int(data[-1,0] + 1)) + '-' + str(int(data[-1,1] % 12 + 1))
    if verbose:
        print ("\nStart date =",start_date)
        print ("End sate =", end_date)
    
    dates = pd.date_range(start_date, end_date, freq='M')
    data_timeseries = pd.Series(data[:,column], index=dates)
    
    if verbose:
        print ("\nTime series data:\n",data_timeseries[:10])
    
    return data_timeseries
    
if __name__=='__main__':
    input_file = 'data_timeseries.txt'
    column_num = 2
    data_timeseries = convert_data_to_timeseries(input_file, column_num)
    data_timeseries.plot()
    plt.title('Input data')
    plt.show()