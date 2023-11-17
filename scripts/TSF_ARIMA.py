from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

data = read_csv('data_daily.csv', header=0, parse_dates=[0], index_col=0)
data.plot()
plt.title("Original Data")
plt.show()

result = adfuller(data['Receipt_Count'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

autocorrelation_plot(data)
plt.title("Original Data")
plt.show()

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(data.Receipt_Count); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(data.Receipt_Count.diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(data.Receipt_Count.diff().diff()); ax3.set_title('2nd Order Differencing')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(data.Receipt_Count, ax=ax1)
plot_acf(data.Receipt_Count.diff().dropna(), ax=ax2)
plot_acf(data.Receipt_Count.diff().diff().dropna(), ax=ax3)
plt.show()

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data.Receipt_Count.diff().dropna())
plt.title("First Difference Partial Autocorrelation")
plt.show()

# p=6, d=1, q=1
model = ARIMA(data.Receipt_Count, order=(6, 1, 1))
model_fit = model.fit()
model_fit.summary()

ax = data.plot()
plot_predict(model_fit, dynamic=False, ax=ax, start=0, end=600)
plt.show()

