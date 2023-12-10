import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.metrics
import datetime

apikey = st.secrets['api_key']
uri_gl = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY\
&symbol=INTC&apikey={apikey}&outputsize=full&datatype=csv'
sns.set_style('darkgrid')

# задаем список таргетов для которых сделали модели
# targets = ['open', 'close', 'high', 'low', 'volume']
targets = ['open', 'close']

@st.cache_data(ttl=60*60*24)
def load_data(uri):
    data = (pd.read_csv(uri).rename({'timestamp': 'date'}, axis=1).sort_values(['date'])\
            .tail(261*7).reset_index(drop=True))
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
    result[f'{target}_prediction'] = result[f'{target}_prediction'].shift(1)

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


MIN_MAX_RANGE = [pd.to_datetime(d_results[selected_target]['date'].min()),
                 pd.to_datetime(d_results[selected_target]['date'].max())]
PRE_SELECTED_DATES = [pd.to_datetime(d_results[selected_target]['date'].min()),
                      pd.to_datetime(d_results[selected_target]['date'].max())]
for i in range(len(MIN_MAX_RANGE)):
    MIN_MAX_RANGE[i] = MIN_MAX_RANGE[i].to_pydatetime()
for i in range(len(PRE_SELECTED_DATES)):
    PRE_SELECTED_DATES[i] = PRE_SELECTED_DATES[i].to_pydatetime()

# Выбор диапазона для построения графика и расчета метрик
values = st.slider(
    'Выберите диапазон времени',
    MIN_MAX_RANGE[0],
    MIN_MAX_RANGE[1],
    PRE_SELECTED_DATES,
    step=datetime.timedelta(days=2),
    format="YYYY-MM-DD",
    )

ind = (pd.to_datetime(d_results[selected_target]['date']).between(
    pd.to_datetime(values[0]),
    pd.to_datetime(values[1]))
)
df_tmp = d_results[selected_target][ind]
# st.write('Values:', values)
# st.write(df_tmp.head())
# st.write(df_tmp.tail())

# построение графика
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='date', y='value', hue='variable',
             data=pd.melt(df_tmp, ['date']),
             palette=['black', 'green'], alpha=0.8, linestyle='--')
t = len(df_tmp) // 10
ax.set_xticks(range(len(df_tmp))[::t], labels=df_tmp.date.values[::t])
plt.grid()
st.pyplot(fig)

# расчет метрик
y_true = df_tmp[selected_target].fillna(df_tmp[selected_target].mean())
y_pred = df_tmp[f'{selected_target}_prediction'].fillna(df_tmp[f'{selected_target}_prediction'].mean())
d_metrics = dict()
d_metrics['MAPE'] = sklearn.metrics.mean_absolute_percentage_error
d_metrics['MAE'] = sklearn.metrics.mean_absolute_error
d_metrics['MedianAE'] = sklearn.metrics.median_absolute_error
for metric in sorted(d_metrics.keys()):
    value = d_metrics[metric](y_true, y_pred)
    st.write(f'{metric} на рассматриваемом периоде:', value)

