import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


model = keras.models.load_model('keras_model.keras')


salesTest = pd.read_csv('test.csv')


salesTest['LotFrontage'].fillna(0, inplace=True)
salesTest['Alley'].fillna('none', inplace=True)
salesTest['FireplaceQu'].fillna('none', inplace=True)
salesTest['Fence'].fillna('none', inplace=True)
salesTest['PoolQC'].fillna('none', inplace=True)
salesTest['MiscFeature'].fillna('none', inplace=True)
salesTest['GarageType'].fillna('none', inplace=True)
salesTest['GarageYrBlt'].fillna(salesTest['GarageYrBlt'].mean(), inplace=True)
salesTest['GarageFinish'].fillna('none', inplace=True)
salesTest['GarageQual'].fillna('none', inplace=True)
salesTest['GarageCond'].fillna('none', inplace=True)
salesTest['MasVnrType'].fillna('none', inplace=True)
salesTest['BsmtQual'].fillna('none', inplace=True)
salesTest['BsmtCond'].fillna('none', inplace=True)
salesTest['BsmtExposure'].fillna('none', inplace=True)
salesTest['BsmtFinType1'].fillna('none', inplace=True)
salesTest['BsmtFinType2'].fillna('none', inplace=True)


def one_hot_encode_and_drop(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
    one_hot_encoded = one_hot_encoded.astype(int)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop(column_name, axis=1)
    return df


categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
                       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

for column in categorical_columns:
    salesTest = one_hot_encode_and_drop(salesTest, column)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(salesTest.drop(['Id'], axis=1))  # Assuming 'Id' is not needed

X_test_scaled = pd.DataFrame(X_test_scaled, columns=salesTest.drop(['Id'], axis=1).columns)

X_test_scaled = sm.add_constant(X_test_scaled)

n_components = 285  
pca = PCA(n_components=n_components)
X_test_pca = pca.fit_transform(X_test_scaled)

y_pred_test = model.predict(X_test_pca)


salesTest['PredictedSalePrice'] = y_pred_test

salesTest[['Id', 'PredictedSalePrice']].to_csv('predicted_test.csv', index=False)









