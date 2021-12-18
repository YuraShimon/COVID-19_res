import numpy as np
import math
import pandas as pd
from prophet import Prophet
import holidays
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.figsize"] = [14, 10]
np.set_printoptions(precision=2, linewidth=500)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)

df = pd.read_csv(r'E:/University/COVID-19/coronavirus-data-master/trends/data-by-day.csv', parse_dates=[0])

list_holidays = []
for date, _ in sorted(holidays.US(state='NY', years=[2020, 2021]).items()):
    if date >= pd.Timestamp('02/29/2020').date():
         list_holidays.append(date)
list_holidays = [x for x in list_holidays if x >= df[df.columns[0]][0] and x <= df[df.columns[0]][df.index[-1]]]
holidays = pd.DataFrame({
  'holiday': 'holiday',
  'ds': list_holidays,
})

predictions = 7

case_count = df[df.columns[0:2]]
case_count.columns = ['ds', 'y']
train_case_count = case_count[:-predictions]
m_case_count = Prophet()
m_case_count.fit(train_case_count)
future_case_count = m_case_count.make_future_dataframe(periods=predictions)
forecast_case_count = m_case_count.predict(future_case_count)
print(forecast_case_count)
case_count_test = case_count['y'][-predictions:].values
case_count_pred = forecast_case_count['yhat'][-predictions:].values
case_count_pred = [math.ceil(case_count_pred[i]) for i in range(len(case_count_pred))]
case_count_pred = [0 if i < 0 else i for i in case_count_pred]

hospitalized_count = df[df.columns[[0, 3]]]
hospitalized_count.columns = ['ds', 'y']
train_hospitalized_count = hospitalized_count[:-predictions]
m_hospitalized_count = Prophet()
m_hospitalized_count.fit(train_hospitalized_count)
future_hospitalized_count = m_hospitalized_count.make_future_dataframe(periods=predictions)
forecast_hospitalized_count = m_hospitalized_count.predict(future_hospitalized_count)
print(forecast_hospitalized_count)
hospitalized_count_test = hospitalized_count['y'][-predictions:].values
hospitalized_count_pred = forecast_hospitalized_count['yhat'][-predictions:].values
hospitalized_count_pred = [math.ceil(hospitalized_count_pred[i]) for i in range(len(hospitalized_count_pred))]
hospitalized_count_pred = [0 if i < 0 else i for i in hospitalized_count_pred]

death_count = df[df.columns[[0, 4]]]
death_count.columns = ['ds', 'y']
train_death_count = death_count[:-predictions]
m_death_count = Prophet()
m_death_count.fit(train_death_count)
future_death_count = m_death_count.make_future_dataframe(periods=predictions)
forecast_death_count = m_death_count.predict(future_death_count)
death_count_test = death_count['y'][-predictions:].values
death_count_pred = forecast_death_count['yhat'][-predictions:].values
death_count_pred = [math.ceil(death_count_pred[i]) for i in range(len(death_count_pred))]
death_count_pred = [0 if i < 0 else i for i in death_count_pred]

fig, axes = plt.subplots(nrows=3, ncols=1)

axes[0].plot(case_count_test, label='Actual CASE_COUNT', color='yellow', linestyle ='-')
axes[0].plot(case_count_pred, label='Predicted CASE_COUNT', color='yellow', linestyle ='--')
axes[1].plot(hospitalized_count_test, label='Actual HOSPITALIZED_COUNT', color='orange', linestyle ='-')
axes[1].plot(hospitalized_count_pred, label='Predicted HOSPITALIZED_COUNT', color='orange', linestyle ='--')
axes[1].set_ylabel('Count')
axes[2].plot(death_count_test, label='Actual DEATH_COUNT', color='black', linestyle='-')
axes[2].plot(death_count_pred, label='Predicted DEATH_COUNT', color='black', linestyle='--')
axes[2].set_xlabel('Day')

leg0 = axes[0].legend(loc='lower left', shadow=False, ncol=1)
leg1 = axes[1].legend(loc='lower left', shadow=False, ncol=1)
leg2 = axes[2].legend(loc='lower left', shadow=False, ncol=1)
plt.show(bbox_inches='tight')
# fig.savefig(r'E:/University/COVID-19/images/predictions/.png')

case_count_errors = [case_count_test[i] - case_count_pred[i] for i in range(len(case_count_test))]
print('case_count_test: ', sum(case_count_test))
print('case_count_pred: ', sum(case_count_pred))
print('case_count Errors: ', sum(case_count_errors))
print(mean_squared_error(case_count_test, case_count_pred, squared=False))

hospitalized_count_errors = [hospitalized_count_test[i] - hospitalized_count_pred[i] for i in range(len(hospitalized_count_test))]
print('hospitalized_count_test: ', sum(hospitalized_count_test))
print('hospitalized_count_pred: ', sum(hospitalized_count_pred))
print('hospitalized_count Errors: ', sum(hospitalized_count_errors))
print(mean_squared_error(hospitalized_count_test, hospitalized_count_pred, squared=False))

death_count_errors = [death_count_test[i] - death_count_pred[i] for i in range(len(death_count_test))]
print('death_count_test: ', sum(death_count_test))
print('death_count_pred: ', sum(death_count_pred))
print('death_count Errors: ', sum(death_count_errors))
print(mean_squared_error(death_count_test, death_count_pred, squared=False))
# from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.model_selection import temporal_train_test_split
# from sktime.forecasting.arima import AutoARIMA
# from sktime.forecasting.theta import ThetaForecaster
# from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
#
# y_train, case_count_test = temporal_train_test_split(case)
# fh = ForecastingHorizon(case_count_test.index, is_relative=False)
# forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
# forecaster.fit(y_train)
# case_count_pred = forecaster.predict(fh)
#
#




