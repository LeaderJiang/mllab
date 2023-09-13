# %%
!pip3 install yfinance
!pip3 install statsmodels
!pip3 install tensorflow


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# %%
# 抓取股票資料
# tickers = ['1301.TW', '1303.TW', '1326.TW', '6505.TW']
tickers = ['GRMN', 'VOO', 'TSLA', 'NVDA']
start_date = '2019-01-01'
end_date = '2023-09-12'
data = yf.download(tickers=tickers, start=start_date, end=end_date)
data


# %%
# 繪製股價圖
plt.figure(figsize=(12, 4))
plt.plot(data.index, data["Adj Close"], label="Price", color="b")
plt.title("Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


# %%
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_scaled



# %%
time_steps = 7
X, y = [], []

for i in range(len(data_scaled) - time_steps):
    X.append(data_scaled[i:i+time_steps])
    y.append(data_scaled[i+time_steps, 0])

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)


# %%
X


# %%
# Train Test Split
train_size = int(0.7 * len(X))
valid_size = int(0.15 * len(X))

X_train, X_valid, X_test = X[:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:]
y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+valid_size], y[train_size+valid_size:]

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)


# %%
model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, 2), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# %%
model.fit(X_train, 
          y_train, 
          epochs=50, 
          batch_size=8, 
          validation_data=(X_valid, y_valid), 
          verbose=1)


# %%
last_data = X[-1].reshape(1, time_steps, 2)
predicted_scaled = model.predict(last_data)[0][0]
predicted_scaled


# %%
predicted_scaled = model.predict(X_test)
predicted_adj_close = scaler.inverse_transform(np.column_stack((predicted_scaled, np.zeros_like(predicted_scaled))))[:, 0]
predicted_adj_close


# %%
y_test_original = scaler.inverse_transform(np.column_stack((y_test, np.zeros_like(y_test))))[:, 0]
y_test_original


# %%
# 绘制实际值与预测值图表
plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size+valid_size+time_steps:], y_test_original, label='True Prices')
plt.plot(data.index[train_size+valid_size+time_steps:], predicted_adj_close, label='Predicted Prices')
plt.title('Actual vs Predicted Adj Close Prices for TSLA')
plt.xlabel('Date')
plt.ylabel('Adj Close Price')
plt.legend()
plt.show()



# %%
# 分析相關係數
# plt.figure(figsize=(12, 10))
# plt.title('Pearson Correlation of Features', y=1.02, size=15)
# sns.heatmap(stock_data['Adj Close'].astype(float).corr(),
#             linewidths=0.1, 
#             square=True, 
#             linecolor='white', 
#             annot=True)
# plt.show()


# %%
# 繪製股價走勢圖
plt.figure(figsize=(16, 9))
plt.plot(data)
plt.title('Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# %%
#找p
plt.figure(figsize=(12, 6))
plot_pacf(data['Adj Close'], lags=30, title='Partial Autocorrelation Function')
plt.show()

#找q
plt.figure(figsize=(12, 6))
plot_acf(data['Adj Close'], lags=35, title='Autocorrelation Function')
plt.show()




# %%
# 準備訓練資料
train_data = data['Adj Close']

# 設定模型參數
p = 2  # AR階數: 取近幾期的資料來做自回歸
d = 0  # 差分階數: 平穩化
q = 34  # MA階數: 
arima = ARIMA(train_data, order=(p, d, q))

# 訓練模型
model = arima.fit()


# %%
# 预测未来一个月的股价
forecast_steps = 1
forecast = model.forecast(steps=forecast_steps, alpha=0.05)
forecast


# %%
forecast_index = pd.date_range(start=train_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='True Prices')
plt.plot(forecast_index, forecast, label='Forecasted Prices')
plt.title('ARIMA Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.show()


# %%
