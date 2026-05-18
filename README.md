# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18:05:2026

### AIM:
To analyze monthly sales trends using time series decomposition and forecast future sales using the Holt-Winters model evaluated by RMSE.


### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/TSLA.csv', parse_dates=['Date'],index_col='Date')

data_monthly = data['Close'].resample('MS').mean()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1,1)).flatten(),
    index=data_monthly.index
)

scaled_data = scaled_data + 1

scaled_data.plot(figsize=(10,5), title='Scaled Data',color='purple')
plt.show()

decomposition = seasonal_decompose(data_monthly, model='additive', period=12)

decomposition.plot()
plt.show()

train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(12,6), label='Train Data',color='green')

test_predictions_add.plot(ax=ax, label='Predictions')
test_data.plot(ax=ax, label='Test Data')

ax.legend()
ax.set_title('Visual Evaluation')
plt.show()

mae = mean_absolute_error(test_data, test_predictions_add)

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

print("MAE :", mae)
print("RMSE :", rmse)

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

final_predictions = final_model.forecast(steps=12)

ax = scaled_data.plot(figsize=(12,6), label='Historical Data')

final_predictions.plot(ax=ax, label='Future Forecast')

ax.legend()
ax.set_title('Future Forecast using Holt-Winters')

ax.set_xlabel('Date')
ax.set_ylabel('Scaled Close Price')

plt.show()

print(final_predictions)

```

### OUTPUT:

<img width="1225" height="613" alt="image" src="https://github.com/user-attachments/assets/be4fa971-221a-4fb4-ba1b-623eddb39764" />
<img width="1063" height="602" alt="image" src="https://github.com/user-attachments/assets/24a68deb-7d32-420a-acf4-3ff99a8ad36f" />
<img width="1407" height="768" alt="image" src="https://github.com/user-attachments/assets/8d5a935c-8596-475b-9498-c8e9acb740e7" />
<img width="1459" height="700" alt="image" src="https://github.com/user-attachments/assets/f6e44a07-2a82-49ae-8e5d-c084ca0b808b" />
<img width="1132" height="323" alt="image" src="https://github.com/user-attachments/assets/e1c6a66d-7d94-44cf-b562-9bacb42c604a" />



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
