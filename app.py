import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime

apikey = st.secrets['api_key']
uri_gl = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY\
&symbol=INTC&apikey={apikey}&outputsize=full&datatype=csv'

# задаем список таргетов для которых сделали модели
# targets = ['open', 'close', 'high', 'low', 'volume']
targets = ['open']

@st.cache_data(ttl=60*60*24)
def load_data(uri):
    data = pd.read_csv(uri).rename({'timestamp': 'date'}, axis=1).sort_values(['date']).reset_index(drop=True)
    return data


@st.cache_data(ttl=60*60*24)
def feature_engineering(data, target):
    data = data[['date', target]]
    # create features for predicting target
    return data


@st.cache_data(ttl=60*60*24)
def create_prediction(data, target):
    model = d_models[target]
    result = data[['date', target]]

    # две строки ниже после реализации моделей
    # features = data.drop([target, 'date'], axis=1)
    # result[f'{target}_prediction'] = model.predict(features)

    # заглушка, пока нет моделей
    result[f'{target}_prediction'] = result[target].copy()

    # ниже менять ничего не надо
    new_date = str(pd.to_datetime(result['date'].max()) + pd.DateOffset(days=1))[:10]
    result.loc[len(result.index)] = [new_date, None, None]
    result[f'{target}_prediction'] = result[f'{target}_prediction'].shift(-1)

    return result

# Загрузка данных
df = load_data(uri_gl)

# чтение предобученных моделей
d_models = dict()
for target in targets:
    # d_models[target] = read model for target
    d_models[target] = None

# создание признаков и предсказание результата
d_results = dict()
for target in targets:
    df_tmp = feature_engineering(df, target)
    d_results[target] = create_prediction(df_tmp, target)


# Выбор таргета
selected_target = st.selectbox('Выберите таргет для построения прогноза', targets)


# Выбор диапазона для построения графика и расчета метрик
MIN_MAX_RANGE = (pd.to_datetime(d_results[selected_target]['date'].min()),
                 pd.to_datetime(d_results[selected_target]['date'].max()))
PRE_SELECTED_DATES = (pd.to_datetime(d_results[selected_target]['date'].min()),
                      pd.to_datetime(d_results[selected_target]['date'].max()))


st.write(d_results[selected_target].head())
st.write(MIN_MAX_RANGE)


values = st.slider(
    'Выберите диапазон времени',
    MIN_MAX_RANGE[0],
    MIN_MAX_RANGE[1],
    PRE_SELECTED_DATES,
    step=datetime.timedelta(days=2),
    format="YYYY-MM-DD",
    )

st.write('Values:', values)
st.write(d_results[selected_target].head())
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.histplot(data[selected_column], kde=True, ax=ax)
# plt.grid()
# st.pyplot(fig)









#
#
# # Загрузка данных
# file_path = 'data.csv'  # Укажите путь к вашему CSV файлу
# data = pd.read_csv(file_path)
#
# # Заголовок приложения
# st.title('Разведочный анализ данных')
#
# # Отображение первых строк данных
# st.subheader('Первые строки данных')
# st.write(data.head())
#
#
# # Описательные статистики
# st.subheader('Описательные статистики')
# st.write(data.describe())
#
# st.subheader('Уникальные значения и их количество')
# l_columns = []
# for col in data.columns:
#     if data[col].nunique() < 10:
#         l_columns.append(col)
# selected_column = st.selectbox('Выберите колонку', l_columns)
# st.write(data[selected_column].value_counts())
#
#
#
# # Графики распределений признаков
# st.subheader('Графики распределений признаков')
#
# # Выберите колонну для построения гистограммы
# selected_column = st.selectbox('Выберите таргет для построения прогноза', targets)
# values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.histplot(data[selected_column], kde=True, ax=ax)
# plt.grid()
# st.pyplot(fig)
#
# # Матрица корреляций
# st.subheader('Матрица корреляций')
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
# st.pyplot(fig)
#
# # Графики зависимостей целевой переменной и признаков
# st.subheader('Scatterplot целевой переменной и признаков')
#
# # Выбираем признак для оси y
# selected_column_y = st.selectbox('Выберите признак для оси y', data.columns)
#
# # Создаем график с осью x как 'TARGET' и осью y как выбранный признак
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x='TARGET', y=selected_column_y, data=data, ax=ax)
# st.pyplot(fig)
#
# # Boxplots для сравнения различных категорий
# st.subheader('Boxplots признаков с делением по TARGET')
#
# # Выбор колонны для сравнения
# selected_column_boxplot = st.selectbox('Выберите колонку для построения Boxplot', data.columns)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.boxplot(x='TARGET', y=selected_column_boxplot, data=data, ax=ax)
# st.pyplot(fig)

# streamlit run main.py