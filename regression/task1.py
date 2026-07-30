import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)


# Split into training and testing sets (80% / 20%)
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardise the Numeric features for train and test data
X_train_numeric = X_train[['age', 'bmi', 'children']]
scaler = StandardScaler().fit(X_train_numeric)
X_train_num_scaled = scaler.transform(X_train_numeric)

X_test_numeric = X_test[['age', 'bmi', 'children']]
X_test_num_scaled = scaler.transform(X_test_numeric)


# preprocess the categorical features for train and test data
X_train_categorical = X_train[['sex', 'smoker', 'region']]
X_test_categorical  = X_test[['sex', 'smoker', 'region']]
encoder = OneHotEncoder(drop='first', sparse_output=False)

encoder.fit(X_train_categorical)
X_train_cat_encoded = encoder.transform(X_train_categorical)
X_test_cat_encoded  = encoder.transform(X_test_categorical)


# combine preprocessed features
X_train_final = np.hstack([X_train_num_scaled, X_train_cat_encoded])
X_test_final  = np.hstack([X_test_num_scaled,  X_test_cat_encoded])


# fit a linear regression 
lin_reg = LinearRegression()
lin_reg.fit(X_train_final, y_train)

# label each coefficient with its corresponding feature name
feature_names = ['age', 'bmi', 'children'] + list(encoder.get_feature_names_out(['sex','smoker','region']))
for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {lin_reg.coef_[i]:.3f}")

# prediction 
lin_reg_prediction = lin_reg.predict(X_test_final)

    
# calculate RMSE
RMSE_train = np.sqrt(mean_squared_error(y_train, lin_reg.predict(X_train_final)))
MAE_train = mean_absolute_error(y_train, lin_reg.predict(X_train_final))
print('Train RMSE: ', round(RMSE_train,3))
print('Train MAE: ', round(MAE_train,3))
RMSE_test = np.sqrt(mean_squared_error(y_test, lin_reg_prediction))
MAE_test = mean_absolute_error(y_test, lin_reg_prediction)
print('Test RMSE: ', round(RMSE_test,3))
print('Test MAE: ', round(MAE_test,3))



# find the range 
minn = min(y_test.min(), lin_reg_prediction.min())
maxx = max(y_test.max(), lin_reg_prediction.max())


# scatter plot of predicted versus actual charges on the test set
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, lin_reg.predict(X_test_final))
ax.plot([minn,maxx],[minn,maxx])
plt.title('Linear Regressions Presiction vs. True Value')
ax.set_xlabel('True Value')
ax.set_ylabel('Prediction')

plt.savefig('Code_task1.png')
plt.show()
