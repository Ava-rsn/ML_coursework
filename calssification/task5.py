import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import pickle
import os
from sklearn.decomposition import PCA



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


# -------------------------------------------------------------------------------
# NOTE:
# The full GridSearchCV (below, now commented out) was run once to select the
# Decision Tree hyperparameters using 5-fold cross-validation on the training set.
# For the final script we simply use the best parameters found earlier and train a single model.
# This avoids re-running the entire grid search every time the script is executed.
# -------------------------------------------------------------------------------

'''
parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", None],
    "splitter": ["best", "random"]
}


decision_tree = DecisionTreeClassifier()

clf = GridSearchCV(
    estimator=decision_tree,
    param_grid=parameters,
    cv=5,
    n_jobs=-1,
    verbose=3
)


start = time.time()
clf.fit(X_train_pca, y_train)

best_model = clf.best_estimator_
duration = time.time() - start
print(f"Decision Tree training time: {duration:.2f} seconds")

print("Best parameters:", clf.best_params_)
print("Cross-validation accuracy: ", clf.best_score_) 

y_pred = best_model.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)


print(f"Training set accuracy: {best_model.score(X_train_pca, y_train)}")
print(f"Test set accuracy: {test_acc}")


base_tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 10, max_features= None, min_samples_leaf= 4, min_samples_split= 2, splitter= 'best')

param_ccp = {
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.01]
}

clf_ccp = GridSearchCV(
    base_tree,
    param_grid=param_ccp,
    cv=5,
    n_jobs=-1,
    verbose=2
)

clf_ccp.fit(X_train_pca, y_train)

best_pruned_tree = clf_ccp.best_estimator_
print("Best ccp_alpha:", clf_ccp.best_params_["ccp_alpha"])

y_pred = best_pruned_tree.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_acc)
'''


# Final model using the best hyperparameters identified through the grid search:
clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 10, max_features= None, min_samples_leaf= 4, min_samples_split= 2, splitter= 'best',ccp_alpha=0.0001)
start = time.time()
clf.fit(X_train_pca, y_train)
duration = time.time() - start
print(f"Decision Tree training time: {duration:.2f} seconds")

y_pred = clf.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
print(f"Training set accuracy: {clf.score(X_train_pca, y_train)}")
print(f"Test set accuracy: {test_acc}")

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

X_test_orig = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

mis_idx = np.where(y_test != y_pred)[0][:5]

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
plt.savefig("DT_misclassified.png")
plt.close()
