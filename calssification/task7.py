import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pickle
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = np.array(batch[b'labels'])
        return X, y
    
def load_cifar10(data_dir):
    X_all = []
    y_all = []

    # 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)

    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)

    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))

    return X_train/255, y_train, X_test/255, y_test


# Load the data
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

pca = PCA(n_components = 200)
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test)


'''
parameters = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10]
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"]
    }
]

svc = SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=3, verbose=3)

start = time.time()
clf.fit(X_train_pca, y_train)
duration = time.time() - start
best_model = clf.best_estimator_

print(f"SVM training time: {duration:.2f} seconds")

print(f"Best hyperparameters: {clf.best_params_}")
print(f"Test set accuracy: {best_model.score(X_test_pca,y_test)}")
print("Best cross-validation accuracy:", clf.best_score_)


y_pred = best_model.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)

print("Training accuracy:", best_model.score(X_train_pca, y_train))
print("Test accuracy:", test_acc)
'''

# GridSearchCV was run once to find the best hyperparameters (see commented block).
# Below we train the SVM using the tuned parameters to avoid re-running the full grid search.

model = SVC(C=10, gamma='scale', kernel='rbf')

start = time.time()
model.fit(X_train_pca, y_train)
duration = time.time() - start
print(f"SVM training time: {duration:.2f} seconds")


y_pred = model.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
print("Training accuracy:", model.score(X_train_pca, y_train))
print("Test accuracy:", test_acc)

X_test_orig = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

mis_idx = np.where(y_test != y_pred)[0][:5]

cifar10_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

plt.figure(figsize=(12, 3))

for i, idx in enumerate(mis_idx):
    img = X_test_orig[idx]
    true_name = cifar10_labels[y_test[idx]]
    pred_name = cifar10_labels[y_pred[idx]]

    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_name}\n Pred: {pred_name}", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("SVM_misclassified.png")
plt.close()
