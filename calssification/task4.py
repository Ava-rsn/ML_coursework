import pickle
import numpy as np
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

print(pca.explained_variance_ratio_.sum()) # 94.6% of the total variance