import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def round_number(df):
    """
    指定された属性の値を四捨五入し整数に置き換える
    """
    ave_columns = ['AveRooms', 'AveBedrms', 'AveOccup']

    for col in ave_columns:
        df[col] = np.round(df[col])

    return df


def std_exclude(df):
    """
    標準偏差の２倍以上の値は取り除く
    """
    columns = df[['MedInc', 'AveRooms', 'Population', 'AveOccup']].columns

    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        boder = np.abs(df[col] - mean) / std
        df = df[(boder < 2)]

    return df


def category(df):
    """
    その区域の人口は、少ない（few）か、普通（usually）か、多い（many）か。
    大体の区域では600人から3000人ということから、この範囲を指標とする。
    """
    if df < 600:
        return 'few'
    elif df > 3000:
        return 'many'
    else:
        return 'usually'


""" 上３つの関数をまとめたカスタム変換器"""
def custom_conversion(dataframe):
    df = dataframe.copy()
    df = round_number(df)

    # サンプルの調査ミスとして取り除く
    df = df[df['HouseAge'] < 52]

    # サンプルの調査ミスとして取り除く
    df = df[df['Price'] < 5]
    df = std_exclude(df)

    # 平均部屋数に対して平均寝室数を比較する
    df['Bedrms_per_Rooms'] = df['AveBedrms'] / df['AveRooms']
    df['Population_Feature'] = df['Population'].apply(category)

    # カテゴリー属性をダミー変数化する
    feature_dummies = pd.get_dummies(df['Population_Feature'], drop_first=True)
    df = pd.concat([df, feature_dummies], axis=1)

    # Xを説明変数、yを目的変数に代入しておく
    X = df.drop(['Price', 'Population_Feature'], axis=1)
    y = df['Price']

    return X, y

def linear(X,y,Xdel_list):
    # 線形モデルで学習
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X) # スケーリング
    lin_reg = LinearRegression()
    lin_reg.fit(X_s, y)

    # モデルの評価
    X_test, y_test = custom_conversion(test_set)
    X_test = X_test.drop(Xdel_list, axis=1)
    X_test = scaler.transform(X_test)
    best_model_pred = lin_reg.predict(X_test)
    best_model_mse = mean_squared_error(y_test, best_model_pred)
    best_model_rmse = np.sqrt(best_model_mse)
    print('テストデータの誤差:{:2f}'.format(best_model_rmse))

sns.set_style('whitegrid')

# データセット
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['Price'] = housing.target
X = df_housing.drop(['Price'], axis=1)
y = df_housing['Price'].copy()

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データをデータフレームに格納
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# データの前処理
X, y = custom_conversion(train_set)
X2 = X.copy()
y2 = y.copy()

# 説明変数の制限
Xdel_list = ["MedInc", "Latitude", "Longitude", "Population", "HouseAge", "many", "usually", "AveBedrms"]
X2del_list = ["MedInc", "Latitude", "Longitude", "Population", "HouseAge", "many", "usually", "Bedrms_per_Rooms"]
X = X.drop(Xdel_list, axis=1)
X2 = X2.drop(X2del_list, axis=1)

linear(X,y,Xdel_list)
linear(X2,y2,X2del_list)