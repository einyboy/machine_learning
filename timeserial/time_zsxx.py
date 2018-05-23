import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.dates as mdate

file_path = 'D:\\DATA_SET\\Times\\ZSXX.csv'
file_path = 'D:\\DATA_SET\\Times\\ZSXX.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv(file_path,  index_col=['DD'], usecols = [1,2],date_parser=dateparse)
data.sort_index(inplace=True)


def test_stationarity(timeseries):
    # �������ͳ��
    rolmean = pd.rolling_mean(timeseries, window=12)    # ��size�����ݽ����ƶ�ƽ��
    rol_weighted_mean = pd.ewma(timeseries, span=12)    # ��size�����ݽ��м�Ȩ�ƶ�ƽ��
    rolstd = pd.rolling_std(timeseries, window=12)      # ƫ��ԭʼֵ����
    # �������ͳ��
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    weighted_mean = plt.plot(rol_weighted_mean, color='green', label='weighted Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # ����df����
    print ('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print (dfoutput)
    
ts = data['CNT']    
plt.plot(ts)
plt.show()
test_stationarity(ts)
plt.show()


# estimating
ts_log = np.log(ts)
# plt.plot(ts_log)
# plt.show()
moving_avg = pd.rolling_mean(ts_log, 12)
# plt.plot(moving_avg)
# plt.plot(moving_avg,color='red')
# plt.show()
ts_log_moving_avg_diff = ts_log - moving_avg
# print ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
plt.show()


# ���differencing
ts_log_diff = ts_log.diff(1)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
plt.show()

ts_log_diff1 = ts_log.diff(1)
ts_log_diff2 = ts_log.diff(2)
ts_log_diff1.plot()
ts_log_diff2.plot()
plt.show()


# �ֽ�decomposing
decomposition = seasonal_decompose(ts)

trend = decomposition.trend  # ����
seasonal = decomposition.seasonal  # ������
plt.plot(seasonal['2018-01-01':'2018-01-31'])
ax = plt.gca()
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#����ʱ���ǩ��ʾ��ʽ
plt.xticks(pd.date_range('2018-01-01', '2018-01-31', freq = '1D'))
plt.gcf().autofmt_xdate()
plt.show()
residual = decomposition.resid  # ʣ���



plt.subplot(411)
plt.plot(ts,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonarity')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.show()




# ȷ������
lag_acf = acf(ts, nlags=20)
lag_pacf = pacf(ts, nlags=20, method='ols')
# q�Ļ�ȡ:ACFͼ�����ߵ�һ�δ�������������.����qȡ2
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')  # lowwer��������
plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')  # upper��������
plt.title('Autocorrelation Function')
# p�Ļ�ȡ:PACFͼ�����ߵ�һ�δ�������������.����pȡ2
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


ts = ts.astype('float')
#AR model
model = ARIMA(ts, order=(5, 0, 0))
result_AR = model.fit(disp=-1)
plt.plot(ts)
plt.plot(result_AR.fittedvalues, color='red')
plt.title('AR model RSS:%.4f' % sum(result_AR.fittedvalues - ts) ** 2)
plt.show()


# MA model
model = ARIMA(ts, order=(0, 0, 2))
result_MA = model.fit(disp=-1)
plt.plot(ts)
plt.plot(result_MA.fittedvalues, color='red')
plt.title('MA model RSS:%.4f' % sum(result_MA.fittedvalues - ts) ** 2)
plt.show()


# ARIMA �������������  Ч������
model = ARIMA(ts, order=(4, 0, 4))
result_ARIMA = model.fit(disp=-1)
plt.plot(ts)
plt.plot(result_ARIMA.fittedvalues, color='red')
plt.title('ARIMA RSS:%.4f' % sum(result_ARIMA.fittedvalues - ts) ** 2)
plt.show()


predictions_ARIMA = pd.Series(result_ARIMA.fittedvalues, copy=True)
# print predictions_ARIMA.head()#����������û�е�һ�е�,��Ϊ��1���ӳ�

plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
plt.show()

#Ԥ��
#predict_sunspots = result_ARIMA.predict('2018-03-10', '2018-05-20', dynamic=True)
predict_sunspots = result_ARIMA.predict('2018-03-10', '2018-05-25')
print(predict_sunspots['2018-05-15':'2018-05-22'])
plt.plot(ts['2018-03-01':])
plt.plot(predict_sunspots)
#plt.xticks(pd.date_range('2018-03-01', '2018-05-20', 5),rotation=90)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#����ʱ���ǩ��ʾ��ʽ
plt.xticks(pd.date_range('2018-03-01', '2018-05-20', freq = '5D'))
plt.gcf().autofmt_xdate()
plt.show()
#�Զ�ȷ������
import statsmodels.api as sm
sm.tsa.arma_order_select_ic(ts,max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC
sm.tsa.arma_order_select_ic(ts,max_ar=6,max_ma=4,ic='bic')['bic_min_order']  # BIC
sm.tsa.arma_order_select_ic(ts,max_ar=6,max_ma=4,ic='hqic')['hqic_min_order'] # HQIC