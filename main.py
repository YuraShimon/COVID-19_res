import numpy as np
import pandas as pd
# import sklearn
# import keras
from sktime.forecasting.naive import NaiveForecaster
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

# import streamlit as st

# Настроювання параметрів для відображення
np.set_printoptions(precision=2, linewidth=500)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Зчитування даних
path1 = r'E:/University/COVID-19/coronavirus-data-master/trends/data-by-day.csv'

data_by_day = pd.read_csv(path1, parse_dates=['date_of_interest'])

# print(data_by_day.dtypes)

#print(data_by_day.head(10))

# fig, ax = plt.subplots(figsize=(32, 8))
#
# ax.plot(data_by_day['date_of_interest'], data_by_day['CASE_COUNT'])


# ax.set_title('Динаміка інфікування за добу')
# ax.set_xlabel('Дата')
# ax.set_ylabel('Число заражених')
# y1 = pd.Timestamp('2020-03-13')
# plt.axvline(y1, color='r', linestyle='--')
# y2 = pd.Timestamp('2020-07-19')
# plt.axvline(y2, color='r', linestyle='--')
# ax.fill_between(data_by_day['date_of_interest'], y1, y2,
#                 facecolor='r',
#                 alpha=0.5)

# path2 = r'E:/University/COVID-19/Weather/2020-2021(1).csv'
#
# weather_data = pd.read_csv(path2, parse_dates=['Date'])


fig1, ax1 = plt.subplots(figsize=(64, 32))  # dpi=1000,

ax1.plot(data_by_day['date_of_interest'], data_by_day['CASE_COUNT'], color='orange', label='CASE_COUNT')
ax1.plot(data_by_day['date_of_interest'], data_by_day['PROBABLE_CASE_COUNT'], color='yellow', label='PROBABLE_CASE_COUNT')
ax1.plot(data_by_day['date_of_interest'], data_by_day['HOSPITALIZED_COUNT'], color='red', label='HOSPITALIZED_COUNT')
ax1.plot(data_by_day['date_of_interest'], data_by_day['DEATH_COUNT'], color='black', label='DEATH_COUNT')
ax1.plot(data_by_day['date_of_interest'], data_by_day['PROBABLE_DEATH_COUNT'], color='grey', label='PROBABLE_DEATH_COUNT')

ax1.legend()
ax1.grid(which='major', color='Black')
ax1.minorticks_on()
ax1.grid(which='minor', linestyle=':', color='gray')
plt.yticks(np.arange(0, 7000, 250))
plt.xticks(rotation=90)
ax1.xaxis_date()
fig1.autofmt_xdate()

plt.show()

# fig2, ax2 = plt.subplots(figsize=(32, 8))
# ax2.plot(weather_data['Date'], weather_data['TempAvg'], color='green')
# ax2.grid(which='major', color='Black')
# ax2.minorticks_on()
# ax2.grid(which='minor', linestyle=':', color='gray')
# plt.xticks(rotation=90)
# ax2.xaxis_date()
# fig2.autofmt_xdate()
#
# plt.show()

