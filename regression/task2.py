import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


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

# reshape y_train and y_test to (N,1) 
y_train_reshaped = y_train.to_numpy().reshape(-1, 1) 
y_test_reshaped = y_test.to_numpy().reshape(-1, 1)


# convert features to float tensors
X_train_tensor = torch.FloatTensor(X_train_final)
X_test_tensor = torch.FloatTensor(X_test_final)
y_train_tensor = torch.FloatTensor(y_train_reshaped)
y_test_tensor  = torch.FloatTensor(y_test_reshaped)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



# my MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_train_final.shape[1],64)
        self.drop1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(64,32)
        self.drop2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(32, 16)
        self.drop3 = nn.Dropout(p=0.05)
        self.out = nn.Linear(16,1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x)) #activation function 
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = F.relu(self.fc3(x))
        x = self.drop3(x)

        x = self.out(x)
        return x


model = MLP()

# set the criterion of model to measure the error
criterion = nn.MSELoss()

# Adam optimizer, learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()  # dropout ON
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # test loss
    model.eval()  # dropout OFF
    with torch.no_grad():
        test_epoch_loss = 0
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            test_epoch_loss += criterion(preds, y_batch).item()

    test_losses.append(test_epoch_loss / len(test_loader))

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")


model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    train_rmse = torch.sqrt(criterion(train_preds, y_train_tensor))
    print("Train RMSE:", float(train_rmse))
    
    test_preds = model(X_test_tensor)
    test_rmse = torch.sqrt(criterion(test_preds, y_test_tensor))
    print("Test RMSE:", float(test_rmse))




plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss Over Epochs")
plt.legend()
# plt.savefig('loss_over_epochs.png')


# convert test y to numpy
y_test_np = y_test_tensor.numpy().flatten()

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).numpy().flatten()

minn = min(test_preds.min(), y_test_np.min())
maxx = max(test_preds.max(), y_test_np.max())

# scatter plot of predicted versus actual charges
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test_np, test_preds)
ax.plot([minn, maxx], [minn, maxx])
plt.title("Neural Network Predictions vs True Value")
plt.xlabel("True Value")
plt.ylabel("Predicted")
plt.savefig('Code_task2.png')
plt.show()
