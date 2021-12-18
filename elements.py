import pandas as pd
import geopandas as gpd

import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'browser'

import holidays

import streamlit as st


# def create_map(df, color, color_cs):
#     figure = px.choropleth(df,
#                         geojson=df.geometry,  # use geometry of df to map
#                         locations=df.index,  # index of df
#                         color=color,  # identify representing column
#                         hover_name='NEIGHBORHOOD_NAME',  # identify hover name
#                         color_continuous_scale=color_cs,
#                         center={'lat': 40.7128, 'lon': 74.0060},
#                         title='',
#                         width=1400,
#                         height=800
#                         )
#     return figure
#
#
# def update_map(figure):
#     figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font_family="Times New Roman", font_size=14)
#     figure.update(layout_showlegend=False)
#     figure.update_geos(fitbounds="locations", visible=False)
#     figure.layout.coloraxis.colorbar.title = ''
#
# gdf = gpd.read_file(r'E:/University/COVID-19/coronavirus-data-master/Geography-resources/MODZCTA_2010.shp')
# gdf = gdf.astype({'MODZCTA': 'int64'})

# df_zcta = pd.read_csv(r'E:/University/COVID-19/coronavirus-data-master/totals/data-by-modzcta.csv')
#
# corrs = df_zcta[df_zcta.columns[[6, 11, 12]]].corr()
# fig = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True, colorscale='earth')
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font_family="Times New Roman", font_size=14)
# fig.write_image(r'E:/University/COVID-19/images/totals/corr.png')

# df_merge_zcta = gdf.merge(df_zcta, right_on='MODIFIED_ZCTA', left_on='MODZCTA')
#
# df_merge_zcta.to_crs('EPSG:4326', inplace=True)
#
# fig = create_map(df_merge_zcta, 'COVID_CONFIRMED_CASE_COUNT', 'oryel')
# update_map(fig)
#
# # st.plotly_chart(fig)
# fig.write_image(r'E:/University/COVID-19/images/data-by-modzcta.CONFIRMED_CASE_COUNT.png')
#
def load_data(path):
    return pd.read_csv(path, parse_dates=[0])
#
# def fig_update(fig, df):
#     fig.update_layout(legend_title_text='')
#     fig.add_vline(x=pd.Timestamp('2020-03-13'), line_width=1, line_dash="dash", line_color="red")
#     fig.add_vline(x=pd.Timestamp('2020-07-19'), line_width=1, line_dash="dash", line_color="red")
#     new_list = [x for x in list_holidays if x >= df[df.columns[0]][0] and x <= df[df.columns[0]][df.index[-1]]]
#     for dt in new_list:
#         fig.add_vline(x=dt, line_width=1, line_dash="dash", line_color="green")
#
# list_holidays = []
# for date, _ in sorted(holidays.US(state='NY', years=[2020, 2021]).items()):
#     if date >= pd.Timestamp('02/29/2020').date():
#          list_holidays.append(date)

# path = r'E:/University/COVID-19/covid-vaccine-data/doses/doses-by-day.csv'
# df = load_data(path)
# df = df.drop(df.columns[[0, 2]], axis=1)

# i_tuple = (20, 21, 22)
# fig = px.line(df, x=df.columns[0], y=(df[df.columns[4]] + df[df.columns[6]]), width=1400, height=800)  # , markers=True
# fig_update(fig, df)
# fig = px.scatter(x=df[df.columns[1]].cumsum(), y=df[df.columns[4]].cumsum(), labels={'x':df.columns[1], 'y':df.columns[4]})
# fig.update_layout(yaxis_title="ADMIN_ALLDOSES_CUMULATIVE",margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=1400, height=800, font_family="Times New Roman", font_size=14)
# fig.write_image(r'E:/University/COVID-19/images/vaccine/doses/ADMIN_ALLDOSES_CUMULATIVE.png')


# branches = ['NON-NYC', 'NYC']
# y1 = df.loc[0,'RACE_ETHNICITY']
# y2 = df.loc[1,'RACE_ETHNICITY']
# y3 = df.loc[2,'RACE_ETHNICITY']
# y4 = df.loc[3,'RACE_ETHNICITY']
# y5 = df.loc[4,'RACE_ETHNICITY']
# y6 = df.loc[5,'RACE_ETHNICITY']
# y7 = df.loc[6,'RACE_ETHNICITY']
# y8 = df.loc[7,'RACE_ETHNICITY']
# y9 = df.loc[8,'RACE_ETHNICITY']
# trace1 = go.Bar(
#    x=df['RACE_ETHNICITY'],
#    y=df['NYC'],
#    name='NYC'
# )
# trace2 = go.Bar(
#    x=df['RACE_ETHNICITY'],
#    y=df['NON-NYC'],
#    name='NON-NYC'
# )
#
# data = [trace1, trace2]  # , margin={"r": 0, "t": 0, "l": 0, "b": 0} trace1, trace2, trace3, trace4
# layout = go.Layout(barmode='group')
# fig = go.Figure(data=data, layout=layout)
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=1400, height=800, font_family="Times New Roman", font_size=14)
# fig.write_image(r'E:/University/COVID-19/images/vaccine/people/by-residency-1plus-allages.png')

# data_by_day = pd.read_csv(r'E:/University/COVID-19/coronavirus-data-master/trends/data-by-day.csv')
# # df = df.drop(df.columns[[0, 1, 3]], axis=1)
# data_by_day = data_by_day.drop(data_by_day.columns[6:], axis=1)
# print(data_by_day.columns)
# weather = pd.read_csv(r'E:/University/COVID-19/Weather/2020-2021.csv')
# weather = weather.drop(weather.columns[[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15]], axis=1)
# print(weather.columns)
# data_by_day = data_by_day.join(weather)
# print(data_by_day.columns)
# corrs = data_by_day.corr()
# fig = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True, colorscale='earth')
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font_family="Times New Roman", font_size=14)
# fig.write_image(r'E:/University/COVID-19/images/trends/weathercorr1.png')



# def invert_boxcox(value, lmbd):
#     if lmbd == 0:
#         return exp(value)
#     return exp(log(lmbd * value + 1) / lmbd)

 # transformed, lmbd = boxcox(df[category1])
    # cov_transformedhist = px.histogram(df, x=df[category1], nbins=10)
    # fig_update(cov_transformedhist)
    # st.plotly_chart(cov_transformedhist)
    #
    # st.write(lmbd)
    #
    # inverted = [invert_boxcox(x, lmbd) for x in transformed]
    # cov_invertedhist = px.histogram(df, x=df[category1], nbins=10)
    # fig_update(cov_invertedhist)
    # st.plotly_chart(cov_invertedhist)

# qqplot_data = sm.qqplot(df[category1], line='45').gca().lines
# fig = go.Figure()
#
# fig.add_trace({
#     'type': 'scatter',
#     'x': qqplot_data[0].get_xdata(),
#     'y': qqplot_data[0].get_ydata(),
#     'mode': 'markers',
#     'marker': {
#         'color': '#19d3f3'
#     }
# })
#
# fig.add_trace({
#     'type': 'scatter',
#     'x': qqplot_data[1].get_xdata(),
#     'y': qqplot_data[1].get_ydata(),
#     'mode': 'lines',
#     'line': {
#         'color': '#636efa'
#     }
#
# })
#
# fig['layout'].update({
#     'title': 'Quantile-Quantile Plot',
#     'xaxis': {
#         'title': 'Theoritical Quantities',
#         'zeroline': False
#     },
#     'yaxis': {
#         'title': 'Sample Quantities'
#     },
#     'showlegend': False,
#     'width': 800,
#     'height': 700,
# })

# cov_graph_cols2[1].plotly_chart(fig, use_container_width=True)


# corrs = df.corr()
# figure = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True, colorscale='earth')
