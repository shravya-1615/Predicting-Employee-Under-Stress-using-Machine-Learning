import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib as jb

#****SELECT YOUR OWN PATH******
data = pd.read_csv('../Dataset/train.csv').dropna(inplace=False)
# convert categorical variables to numerical
data = data.drop(['Employee ID', 'Date of Joining'], axis=1, inplace=False)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Company Type'] = data['Company Type'].map({'Service': 0, 'Product': 1})
data['WFH Setup Available'] = data['WFH Setup Available'].map({'No': 0, 'Yes': 1})


# split the data into training and testing sets
X = data.drop('Burn Rate', axis=1)
y = data['Burn Rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# define the parameters for the LightGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['mse', 'mae'],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# train the LightGBM model
model = lgb.train(
    params, train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    early_stopping_rounds=50
)

# make predictions using the trained model
y_pred = model.predict(X_test)

# evaluate the performance of the model
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

jb.dump(model,open("C:/Users/Admin/Contacts/Desktop/project/lightgbm_test/Model/lightgbm.bin","wb"))