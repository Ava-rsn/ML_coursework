import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pymc as pm


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
X_test_categorical = X_test[['sex', 'smoker', 'region']]
encoder = OneHotEncoder(drop='first', sparse_output=False)

encoder.fit(X_train_categorical)
X_train_cat_encoded = encoder.transform(X_train_categorical)
X_test_cat_encoded = encoder.transform(X_test_categorical)


# combine preprocessed features
X_train_final = np.hstack([X_train_num_scaled, X_train_cat_encoded])
X_test_final = np.hstack([X_test_num_scaled, X_test_cat_encoded])

feature_names = ['age', 'bmi', 'children'] + list(encoder.get_feature_names_out(['sex','smoker','region']))



# start a PyMC model
model = pm.Model()

y_mean = y_train.mean()
y_sd = y_train.std()
print(f"y_std : {y_sd}")
print(f"y_mean : {y_mean}")


with model:
    # Defining the priors
    alpha = pm.Normal('alpha', mu=y_mean, sigma=2*y_sd)
    beta = pm.Normal('beta', mu=0, sigma=y_sd, shape= X_train_final.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=y_sd)

    mu = alpha + pm.math.dot(X_train_final, beta)

    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_train)
    # inference
    sampler = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    idata = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.95,
        chains=4,
        cores=1,
        random_seed=42
    )


posterior = idata.posterior


alpha_mean = posterior["alpha"].mean(dim=("chain", "draw")).item()
beta_means = posterior["beta"].mean(dim=("chain", "draw")).values
beta_means_r = np.round(beta_means, 3)
sigma_mean = posterior["sigma"].mean(dim=("chain", "draw")).item()
print("\nPosterior means:")
print(f"alpha: {round(alpha_mean,3)}")
print(f"sigma: {round(sigma_mean, 3)}")


for name, coef in zip(feature_names, beta_means_r):
    print(f"{name}:{coef}")


# posterior prediction on test set
# flatten (chain, draw) -> (sample)
alpha_samples = posterior["alpha"].stack(sample=("chain", "draw")).values
beta_samples = posterior["beta"].stack(sample=("chain", "draw")).values
beta_samples = beta_samples.T


X_test_arr = np.asarray(X_test_final) # shape (n_test, n_features)
y_test_arr = np.asarray(y_test)

print("alpha_samples:", alpha_samples.shape)
print("beta_samples:", beta_samples.shape)
print("X_test_arr:", X_test_arr.shape)


# compute model predictions for each posterior sample:
# y_s = alpha_s + X_test.beta_s
y_pred_samples = alpha_samples[:, None] + beta_samples @ X_test_arr.T # (n_samples, n_test)

# take posterior predictive mean
y_pred_mean = y_pred_samples.mean(axis=0)

# metrics
rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_mean))
mae = mean_absolute_error(y_test_arr, y_pred_mean)
r2 = r2_score(y_test_arr, y_pred_mean)

print("\nTest set performance:")
print(f" RMSE: {rmse:.3f}")
print(f" MAE: {mae:.3f}")
print(f" R^2: {r2:.3f}")
