from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LSTM, Dense, Dropout
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numpy as np
import pandas as pd
from finta import TA
import seaborn as sns
from matplotlib import pyplot as plt
import glob


def modify_stock_data(df, stock='SBER'):
    # Отбираем нужные акции по ID и готовим датасет.
    # Добавляем индикаторы на основе finta.
    print(df.info())
    print(df.head())
    df['date'] = pd.to_datetime(df['date'])

    data = df[df['full_name'] == f'MOEX:{stock}'].iloc[:, 4:]
    print(data)
    data.set_index('date', inplace=True)
    # print(data.info())
    data['RSI'] = TA.RSI(data)
    data['SMA'] = TA.SMA(data, 12)
    macd = TA.MACD(data)
    data['MACD'] = macd.MACD
    data['SIGNAL'] = macd.SIGNAL
    data['EMA'] = TA.EMA(data)
    rdf = data.dropna()
    print(rdf.head())
    print(rdf.info())

    return rdf.iloc[:, 0:11]


def normalize_data(data):
    # При помощи boxcox выполним преобразование Бокса-Кокса.
    # Для более равномерного распределения данных
    lamb = []
    for i in data.columns:
        print(i)
        if i not in ['MACD', 'SIGNAL', 'volume']:
            data[i], l = boxcox(data[i])
            lamb.append(l)
    # Стандартизация данных датасета.
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    scaler = scaler.fit(data)
    df_to_train = scaler.transform(data)

    # Возвращаем подготовленные данные, и коэффициенты преобразований для обратной трансформации
    return df_to_train, scaler, lamb


def find_file(direct, file_name):
    p_file = glob.glob(f'{direct}{file_name}')
    return p_file


def plot_graf(df, list_indicate=[]) -> None:
    plt.figure(figsize=(8, 6))
    for i in list_indicate:
        sns.lineplot(data=df, x=df.index, y=df[i])
    plt.show()


def get_train_data(df, day_f, day_p):
    train_x = []
    train_y = []
    # Подготовка данных обучающей выборки.
    # Берем данные по размеру окна day_p добавляем их train_x,
    # это данные за определенный период времени допустим с 1 по 10 день. На них модель будет пытаться прогнозировать,
    # значаения которые мы добавляем в train_y. Это кол-во дней из day_f следующие за размером окна. Допустим 11 день.
    for i in range(day_p, len(df) - day_f + 1):
        train_x.append(df[i - day_p:i, 0:df.shape[1]])
        train_y.append(df[i + day_f - 1: i + day_f, 0])
    return np.array(train_x), np.array(train_y)


def model_create(x, y, n_q, model_id=0, epoch=50, b_size=32):
    model = Sequential()
    # Формируем модель, которая будет обучаться на выборке.
    #
    if model_id == 1:
        model.add(LSTM(11, activation='relu', input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
        model.add(LSTM(5, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(train_y.shape[1]))

    elif model_id == 2:
        node = 50
        if n_q != 0:
            node = n_q
        model.add(LSTM(node, return_sequences=False, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))

    elif model_id == 3:
        node = 20
        if n_q != 0:
            node = n_q
        model.add(LSTM(node, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(node))
        model.add(Dropout(0.2))
        model.add(Dense(1))

    else:
        # Используется эта модель.
        # Она состоит из трех слоев LSTM
        # Трех слоев Dropout, которые через определенное время сбрасываю некоторые значения до 0,
        # чтобы не было переобучения модели.
        # И Выходной слой с одним значением.
        node = 30
        if n_q != 0:
            node = n_q
        model.add(LSTM(node, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(node, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(node))
        model.add(Dropout(0.2))
        model.add(Dense(1))

    # Компилируем модель, добавляя метрики
    model.compile(optimizer='adam', metrics=['mse', 'mae'], loss='mse')
    model.summary()

    # Обучаем модель с установленными параметрами.
    hist = model.fit(x, y, epochs=epoch, batch_size=b_size, validation_split=0.2, verbose=1)
    return model, hist


if __name__ == '__main__':

    STOCK = 'LKOH'
    DIR_PKL = 'data/pkl/'
    DAY_FUTURE = 1
    DAY_PAST = 20
    NODE_QUANT = 35  # 35
    M_ID = 3  # 3
    EPOCH = 37  # 40

    # Читаем данные из фаила, если данные были подготовлены ранее, то берем подготовленные
    if find_file(DIR_PKL, f'{STOCK}_tr.pkl'):
        df = pd.read_pickle(f'{DIR_PKL}{STOCK}_tr.pkl')
        print('File Read from PKL')
    else:
        data = pd.read_pickle('data/StockData/BLUE_1D.pkl')
        df = modify_stock_data(data, STOCK)
        df.to_pickle(f'{DIR_PKL}{STOCK}_tr.pkl')

    plot_graf(df.loc[:], ['open'])

    # train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    df_for_train, scaler, l_d = normalize_data(df.loc[:])
    train_x, train_y = get_train_data(df_for_train, DAY_FUTURE, DAY_PAST)

    print(train_x.shape)
    print(train_y.shape)

    model, history = model_create(train_x, train_y, n_q=NODE_QUANT, model_id=M_ID, epoch=EPOCH, b_size=32)

    # Выводим графики метрик обучения модели и записываем его в фаил.
    f, ax = plt.subplots(figsize=(8, 6))
    l1, = ax.plot(history.epoch, history.history['loss'])
    l2, = ax.plot(history.epoch, history.history['mse'])
    l3, = ax.plot(history.epoch, history.history['mae'])
    l4, = ax.plot(history.epoch, history.history['val_mse'])
    l5, = ax.plot(history.epoch, history.history['val_mae'])

    plt.legend((l2, l3, l4, l5), ('mse', 'MAE', 'val_mse', 'val_mae'))
    plt.title(f'Показатели обучения, model:{M_ID} window:{DAY_PAST}, epoch:{EPOCH}, node:{NODE_QUANT}')
    plt.savefig(f'Graf_Screen/MSE_{M_ID}_{EPOCH}_{NODE_QUANT}_{DAY_PAST}')
    plt.show()


    ret_d = 150
    n_day_past = train_x.shape[0] - ret_d
    test = train_x[n_day_past:]

    # print(train_x_t)
    # Делаем прогноз на основе данных прошлых 120 дней.
    # И обученной модели. Сравниваем их с реальными данными.
    #
    pr_t = model.predict(test)
    pr_copy_t = np.repeat(pr_t, df_for_train.shape[1], axis=-1)
    pred_test = scaler.inverse_transform(pr_copy_t)[:, 0]
    pred_test = inv_boxcox(pred_test, l_d[0])
    data_plot = df.iloc[n_day_past+DAY_PAST:df_for_train.shape[0], 0].reset_index()
    data_plot['predict'] = pred_test

    f, ax = plt.subplots(figsize=(8, 6))
    s1, = ax.plot(data_plot['date'], data_plot['open'])
    s2, = ax.plot(data_plot['date'], data_plot['predict'])
    plt.legend((s1, s2), ('original', 'predict'))
    plt.title(f'Прогнозирование движения цены {ret_d}_id:{M_ID}_ep:{EPOCH}_n:{NODE_QUANT}')
    plt.savefig(f'Graf_Screen/CHART_PREDICT_m{M_ID}_ep{EPOCH}_n{NODE_QUANT}')
    plt.show()

    n_day_predict = 80
    train_x_t = train_x[-n_day_predict:]
    # print(train_x_t)

    pr = model.predict(train_x_t)
    pr_copy = np.repeat(pr, df_for_train.shape[1], axis=-1)
    y_pred = scaler.inverse_transform(pr_copy)[:, 0]
    y_pred = inv_boxcox(y_pred, l_d[0])

    pred_days_date = pd.date_range(list(df.index)[-n_day_predict+DAY_PAST], periods=n_day_predict, freq='1d').to_list()

    df_pred = pd.DataFrame({'date': pd.to_datetime(pred_days_date), 'predict': y_pred})
    original = df.loc['20220101':, 'open']
    original = original.reset_index()

    f, ax = plt.subplots(figsize=(8, 6))
    l1, = ax.plot(original['date'], original['open'])
    l2, = ax.plot(df_pred['date'], df_pred['predict'])
    plt.legend((l1, l2), ('real_price', 'predict'))
    plt.ylabel('Price')
    plt.title(f'Прогноз графика цены акций, на основе данных предыдущих {n_day_predict} дней.')
    plt.savefig(f'Graf_Screen/CHART_PREDICT_{n_day_predict}')
    plt.show()


