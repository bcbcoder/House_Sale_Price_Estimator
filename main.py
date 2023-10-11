import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

tf.random.set_seed(42)

print(tf.__version__)
salesTrain = pd.read_csv('train.csv')
salesTest = pd.read_csv('test.csv')


salesTrain['LotFrontage'].fillna(0, inplace=True)
salesTrain['Alley'].fillna('none', inplace=True)
salesTrain['FireplaceQu'].fillna('none', inplace=True)
salesTrain['Fence'].fillna('none', inplace=True)
salesTrain['PoolQC'].fillna('none', inplace=True)
salesTrain['MiscFeature'].fillna('none', inplace=True)
salesTrain['GarageType'].fillna('none', inplace=True)
salesTrain['GarageYrBlt'].fillna(salesTrain['GarageYrBlt'].mean(), inplace=True)
salesTrain['GarageFinish'].fillna('none', inplace=True)
salesTrain['GarageQual'].fillna('none', inplace=True)
salesTrain['GarageCond'].fillna('none', inplace=True)
salesTrain['MasVnrType'].fillna('none', inplace=True)
salesTrain['BsmtQual'].fillna('none', inplace=True)
salesTrain['BsmtCond'].fillna('none', inplace=True)
salesTrain['BsmtExposure'].fillna('none', inplace=True)
salesTrain['BsmtFinType1'].fillna('none', inplace=True)
salesTrain['BsmtFinType2'].fillna('none', inplace=True)
salesTrain.dropna(inplace=True)



def one_hot_encode_and_drop(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
    one_hot_encoded = one_hot_encoded.astype(int)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop(column_name, axis=1)
    return df


salesTrain = one_hot_encode_and_drop(salesTrain, 'MSZoning')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Street')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Alley')
salesTrain = one_hot_encode_and_drop(salesTrain, 'LotShape')
salesTrain = one_hot_encode_and_drop(salesTrain, 'LandContour')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Utilities')
salesTrain = one_hot_encode_and_drop(salesTrain, 'LotConfig')
salesTrain = one_hot_encode_and_drop(salesTrain, 'LandSlope')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Neighborhood')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Condition1')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Condition2')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BldgType')
salesTrain = one_hot_encode_and_drop(salesTrain, 'HouseStyle')
salesTrain = one_hot_encode_and_drop(salesTrain, 'RoofStyle')
salesTrain = one_hot_encode_and_drop(salesTrain, 'RoofMatl')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Exterior1st')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Exterior2nd')
salesTrain = one_hot_encode_and_drop(salesTrain, 'MasVnrType')
salesTrain = one_hot_encode_and_drop(salesTrain, 'ExterQual')
salesTrain = one_hot_encode_and_drop(salesTrain, 'ExterCond')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Foundation')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BsmtQual')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BsmtCond')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BsmtExposure')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BsmtFinType1')
salesTrain = one_hot_encode_and_drop(salesTrain, 'BsmtFinType2')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Heating')
salesTrain = one_hot_encode_and_drop(salesTrain, 'HeatingQC')
salesTrain = one_hot_encode_and_drop(salesTrain, 'CentralAir')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Electrical')
salesTrain = one_hot_encode_and_drop(salesTrain, 'KitchenQual')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Functional')
salesTrain = one_hot_encode_and_drop(salesTrain, 'FireplaceQu')
salesTrain = one_hot_encode_and_drop(salesTrain, 'GarageType')
salesTrain = one_hot_encode_and_drop(salesTrain, 'GarageFinish')
salesTrain = one_hot_encode_and_drop(salesTrain, 'GarageQual')
salesTrain = one_hot_encode_and_drop(salesTrain, 'GarageCond')
salesTrain = one_hot_encode_and_drop(salesTrain, 'PavedDrive')
salesTrain = one_hot_encode_and_drop(salesTrain, 'PoolQC')
salesTrain = one_hot_encode_and_drop(salesTrain, 'Fence')
salesTrain = one_hot_encode_and_drop(salesTrain, 'MiscFeature')
salesTrain = one_hot_encode_and_drop(salesTrain, 'SaleType')
salesTrain = one_hot_encode_and_drop(salesTrain, 'SaleCondition')





y = salesTrain['SalePrice']
X = salesTrain.drop(['SalePrice', 'Id'], axis=1)  # Assuming 'SalePrice' is the dependent variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled = sm.add_constant(X_scaled)


n_components = 285  # Choose the number of components you want to keep
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)


model = keras.Sequential([
    layers.Input(shape=(n_components,)),  # Input layer with the number of PCA components
    layers.Dense(64, activation='relu'),  # Add hidden layers as needed
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

y = y.to_numpy()

# Initialize KFold cross-validator
kf = KFold(n_splits=400, shuffle=True, random_state=42)

all_train_losses = []
all_test_losses = []

for train_index, test_index in kf.split(X_pca):
    X_train_cv, X_test_cv = X_pca[train_index], X_pca[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]

model.fit(X_pca, y, epochs=100, batch_size=32, verbose=1)

y_pred = model.predict(X_pca)

mae = mean_absolute_error(y, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

for prediction in y_pred[:10]:
    print(prediction)
