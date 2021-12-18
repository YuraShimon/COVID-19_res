import os
import sys
import math

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.figsize"] = [14, 10]
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
import holidays
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'browser'

import streamlit as st

@st.cache()
def load_data(path):
    return pd.read_csv(path, parse_dates=[0])


@st.cache()
def modify_cartographic_df(path, gdf, right_on, left_on, crs):
    c_df = pd.read_csv(path)
    df_merge_zcta = gdf.merge(c_df, right_on=right_on, left_on=left_on)
    df_merge_zcta.to_crs(crs, inplace=True)
    return df_merge_zcta


def fig_update(fig):
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.add_vline(x=pd.Timestamp('2020-03-13'), line_width=0.5, line_dash="dash", line_color="red")
    fig.add_vline(x=pd.Timestamp('2020-07-19'), line_width=0.5, line_dash="dash", line_color="red")


def familiarization(df):
    st.write('Фрагмент даних:')
    # cols = st.columns(2)
    # cols[0]\
    st.write(df.head(25), use_container_width=True)
    # st.write(df.info(verbose=True))
    # cols[1]\
    st.write('Статистичні величини, що описують вибірку:')
    st.write(df.describe(), use_container_width=True)


def read_df(file):
    extension = file.name.split('.')[1]
    if extension.upper() == 'CSV':
        st.write(str(file))
        df = load_data(file)
        return df
    elif extension.upper() == 'XLSX':
        df = pd.read_excel(file, engine='openpyxl')
        return df
    elif extension.upper() == 'txt':
        pass


def create_map(df, color, color_cs):
    figure = px.choropleth(df,
                        geojson=df.geometry,  # use geometry of df to map
                        locations=df.index,  # index of df
                        color=color,  # identify representing column
                        hover_name='NEIGHBORHOOD_NAME',  # identify hover name
                        color_continuous_scale=color_cs,
                        center={'lat': 40.7128, 'lon': 74.0060},
                        title='',
                        width=350,
                        height=200
                        )
    return figure


def update_map(figure):
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font_family="Times New Roman", font_size=14)
    figure.update(layout_showlegend=False)
    figure.update_geos(fitbounds="locations", visible=False)
    figure.layout.coloraxis.colorbar.title = ''


st.set_page_config(layout="wide")


sidebar_subheader = st.sidebar.subheader('Інформація щодо використання сервісу')
sidebar_text = st.sidebar.write('Дані та функціональні елементи згруповні тематично у розділах:<br>'
                                '1)"Оцінювання історичних показників пандемії",<br>'
                                '2)"Дослідження взаєзмозв\'язків із факторами середовища",<br>'
                                '3)"Прогнозування розвитку епідеміологічної ситуації".<br>'
                                '<br>'
                                'Щоб ознайомитися з різними показниками пандемії, оберіть <br>'
                                'необхідний файл із запропонованого списку та натисніть на кнопку <br>'
                                '"Browse files" (обмеження щодо розміру та типу підтримуваних файлів <br>'
                                'вказані у полі завантаження).<br>'
                                '<br>'
                                'Для ознайомлення з динамікою іншого показника достатньо вибрати<br>'
                                'необхідний із відповідного випадаючого списка.<br>'
                                '<br>'
                                'Взаємодіючи з інтерактивними графіками, можна змінювати їхній розмір, встановлювати "вікно"<br>'
                                'перегляду, завантажувати статичне зображення, змінювати <br>'
                                'деталізацію інформації.<br>'
                                '<br>'
                                'Для отримання прогнозу слід обрати відповідний часовий інтервал,<br>'
                                'і натиснути на кнопку "Здійснити прогнозування."', unsafe_allow_html=True)

st.title('Аналіз і прогнозування розвитку COVID-19 у місті Нью-Йорк')
st.header('Оцінювання історичних показників пандемії')  # Зведені дані розвитку хвороби за період із 29.02.20 по 13.11.21

file1 = st.file_uploader("Завантажити файл", type=['csv', 'xlsx', 'txt'], accept_multiple_files=False)

if not file1:
    st.write("Завантажте .csv, .xlsx чи .txt файл")
else:
    df = read_df(file1)
    familiarization(df)

    cat_list1 = df.columns[1:]  # data_by_day
    category1 = st.selectbox("Оберіть епідеміологічний показник для відображення:", cat_list1)

    # cov_graph_cols1 = st.columns(2)

    cov_ts = px.line(df, x=df.columns[0], y=df[category1])  # data_by_day  # category1 # , markers=True
    fig_update(cov_ts)
    st.plotly_chart(cov_ts, use_container_width=True)
    # cov_graph_cols1[0]
    cov_graph_cols2 = st.columns(2)


    cov_cumulative = px.line(df, x=df.columns[0], y=df[category1].cumsum())  # data_by_day  # category1 # , markers=True
    fig_update(cov_cumulative)
    cov_graph_cols2[0].plotly_chart(cov_cumulative, use_container_width=True)

    cov_hist = px.histogram(df, x=df[category1], nbins=10)  # data_by_day  # category1 # , markers=True
    fig_update(cov_hist)
    cov_graph_cols2[1].plotly_chart(cov_hist, use_container_width=True)


    gdf = gpd.read_file(r'./data/coronavirus-data-master/Geography-resources/MODZCTA_2010.shp')
    # st.write(gdf.head())
    gdf = gdf.astype({'MODZCTA': 'int64'})

    df_merge_zcta = modify_cartographic_df(r'./data/coronavirus-data-master/totals/data-by-modzcta.csv',
                                           gdf, 'MODIFIED_ZCTA', 'MODZCTA', 'EPSG:4326')


    # df_vaccine_zcta = r'E:/University/COVID-19/covid-vaccine-data/people/coverage-by-modzcta-allages.csv'
    # # df_vaccine_zcta = df_vaccine_zcta.astype({'MODZCTA': 'int64'})
    # df_vaccine_zcta.to_crs('EPSG:4326', inplace=True)

    # st.subheader('Територіальний розподіл будується згідно з системою MODZCTA')
    cov_map_cols1 = st.columns(2)

    cov_map_cols1[0].subheader(
        'Територіальний розподіл населення міста Нью-Йорк станом на 2010 рік')
    map1 = create_map(df_merge_zcta, 'POP_DENOMINATOR', 'blues')
    update_map(map1)
    cov_map_cols1[0].plotly_chart(map1, use_container_width=True)

    cov_map_cols1[1].subheader(
        'Територіальний розподіл інфікованих станом на 13.11.2021')
    map2 = create_map(df_merge_zcta, 'COVID_CONFIRMED_CASE_COUNT', 'oryel')
    update_map(map2)
    cov_map_cols1[1].plotly_chart(map2, use_container_width=True)

    cov_map_cols1[0].subheader(
        'Територіальний розподіл померлих станом на 13.11.2021')
    map3 = create_map(df_merge_zcta, 'COVID_CONFIRMED_DEATH_COUNT', 'gray')
    update_map(map3)
    cov_map_cols1[0].plotly_chart(map3, use_container_width=True)

    # cov_map_cols1[1].subheader(
    #     'Територіальний розподіл вакцинованих станом на 13.11.2021')
    # map4 = create_map(df_vaccine_zcta, 'COUNT_FULLY_CUMULATIVE', 'speed')
    # update_map(map4)
    # cov_map_cols1[1].plotly_chart(map4, use_container_width=True)

    st.write('-' * 500)

    # cat_list1 = st.multiselect('Оберіть показники для відображення', df.columns[1:])
    # draw1 = st.button('Відобразити')
    # if draw1:
    #     new_df = df[df.columns.isin(cat_list1)]
    #     st.write(new_df)

# # path1 = r'E:/University/COVID-19/coronavirus-data-master/trends/data-by-day.csv'
# # data_by_day = load_data(path1)

path2 = r'./data/Weather/2020-2021.csv'
weather_data = load_data(path2)

st.header('Дослідження взаєзмозв\'язків із факторами середовища')
cat_list2 = weather_data.columns[1:]
category2 = st.selectbox("Оберіть метеорологічний показник для відображення:", cat_list2)

fig2 = px.line(weather_data, x=weather_data['Date'], y=weather_data[category2])  # , markers=True
fig_update(fig2)
weather_cols1 = st.columns(2)
weather_cols1[0].plotly_chart(fig2, use_container_width=True)
weather_cols1[1].image(r'./data/images/trends/weathercorr1.png', use_column_width=True)

df = pd.read_csv(r'./data/coronavirus-data-master/trends/data-by-day.csv', parse_dates=[0])

list_holidays = []
for date, _ in sorted(holidays.US(state='NY', years=[2020, 2021]).items()):
    if date >= pd.Timestamp('02/29/2020').date():
         list_holidays.append(date)
list_holidays = [x for x in list_holidays if x >= df[df.columns[0]][0] and x <= df[df.columns[0]][df.index[-1]]]
holidays = pd.DataFrame({
  'holiday': 'holiday',
  'ds': list_holidays,
})

st.header('Прогнозування розвитку епідеміологічної ситуації')
predictions = st.radio("Оберіть період для прогнозування (у днях)", ('7', '14', '30', '45', '60'))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
if st.button('Здійснити пронозування'):
    case_count = df[df.columns[0:2]]
    case_count.columns = ['ds', 'y']
    train_case_count = case_count[:-int(predictions)]
    m_case_count = Prophet()
    m_case_count.fit(train_case_count)
    future_case_count = m_case_count.make_future_dataframe(periods=int(predictions))
    forecast_case_count = m_case_count.predict(future_case_count)
    print(forecast_case_count)
    case_count_test = case_count['y'][-int(predictions):].values
    case_count_pred = forecast_case_count['yhat'][-int(predictions):].values
    case_count_pred = [math.ceil(case_count_pred[i]) for i in range(len(case_count_pred))]
    case_count_pred = [0 if i < 0 else i for i in case_count_pred]

    hospitalized_count = df[df.columns[[0, 3]]]
    hospitalized_count.columns = ['ds', 'y']
    train_hospitalized_count = hospitalized_count[:-int(predictions)]
    m_hospitalized_count = Prophet()
    m_hospitalized_count.fit(train_hospitalized_count)
    future_hospitalized_count = m_hospitalized_count.make_future_dataframe(periods=int(predictions))
    forecast_hospitalized_count = m_hospitalized_count.predict(future_hospitalized_count)
    print(forecast_hospitalized_count)
    hospitalized_count_test = hospitalized_count['y'][-int(predictions):].values
    hospitalized_count_pred = forecast_hospitalized_count['yhat'][-int(predictions):].values
    hospitalized_count_pred = [math.ceil(hospitalized_count_pred[i]) for i in range(len(hospitalized_count_pred))]
    hospitalized_count_pred = [0 if i < 0 else i for i in hospitalized_count_pred]

    death_count = df[df.columns[[0, 4]]]
    death_count.columns = ['ds', 'y']
    train_death_count = death_count[:-int(predictions)]
    m_death_count = Prophet()
    m_death_count.fit(train_death_count)
    future_death_count = m_death_count.make_future_dataframe(periods=int(predictions))
    forecast_death_count = m_death_count.predict(future_death_count)
    death_count_test = death_count['y'][-int(predictions):].values
    death_count_pred = forecast_death_count['yhat'][-int(predictions):].values
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
    st.write(fig)
    # fig.savefig(r'E:/University/COVID-19/images/predictions/.png')
    cov_prediction_cols2 = st.columns(3)
    case_count_errors = [case_count_test[i] - case_count_pred[i] for i in range(len(case_count_test))]
    cov_prediction_cols2[0].write('**Кількість нових випадків**')
    cov_prediction_cols2[0].write('Реальна кількість:' + str(sum(case_count_test)))
    cov_prediction_cols2[0].write('Прогнозована кількість:' + str(sum(case_count_pred)))
    cov_prediction_cols2[0].write('Різниця:' + str(sum(case_count_errors)))
    cov_prediction_cols2[0].write('Середньоквадратична помилка: ' + str(mean_squared_error(case_count_test, case_count_pred, squared=False)))

    hospitalized_count_errors = [hospitalized_count_test[i] - hospitalized_count_pred[i] for i in range(len(hospitalized_count_test))]
    cov_prediction_cols2[1].write('**Кількість госпіталізованих**')
    cov_prediction_cols2[1].write('Реальна кількість: ' + str(sum(hospitalized_count_test)))
    cov_prediction_cols2[1].write('Прогнозована кількість: ' + str(sum(hospitalized_count_pred)))
    cov_prediction_cols2[1].write('Різниця: ' + str(sum(hospitalized_count_errors)))
    cov_prediction_cols2[1].write('Середньоквадратична помилка: ' + str(mean_squared_error(hospitalized_count_test, hospitalized_count_pred, squared=False)))

    death_count_errors = [death_count_test[i] - death_count_pred[i] for i in range(len(death_count_test))]
    cov_prediction_cols2[2].write('**Кількість летальних випадків**')
    cov_prediction_cols2[2].write('Реальна кількість: ' + str(sum(death_count_test)))
    cov_prediction_cols2[2].write('Прогнозована кількість : ' + str(sum(death_count_pred)))
    cov_prediction_cols2[2].write('Різниця: ' + str(sum(death_count_errors)))
    cov_prediction_cols2[2].write('Середньоквадратична помилка: ' + str(mean_squared_error(death_count_test, death_count_pred, squared=False)))




