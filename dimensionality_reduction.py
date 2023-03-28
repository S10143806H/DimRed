# Import necessary package sklearn
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import time

# Load high-dimensional dataset
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_test.data"
data = pd.read_csv(url, header=None)
print(data)


# Preprocessing and normalization
X = data.iloc[:, :-1].values  # iloc integer-location based indexing
y = data.iloc[:, -1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model on original dataset
start_time = time.time()
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred)
original_time = time.time() - start_time

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Apply LDA for Dimensionality Reduction
lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Mahalanobis distance classifier


def mahalanobis_classifier(X_train, X_test, y_train):
    unique_labels = np.unique(y_train)
    means = {}
    cov_inv = np.linalg.inv(np.cov(X_train.T))

    for label in unique_labels:
        means[label] = np.mean(X_train[y_train == label], axis=0)

    predictions = []
    for x in X_test:
        distances = {label: mahalanobis(
            x, means[label], cov_inv) for label in unique_labels}
        min_distance_label = min(distances, key=distances.get)
        predictions.append(min_distance_label)

    return predictions


# # Classification and accuracy computation
# start_time = time.time()

# lr_pca = LogisticRegression(random_state=42)
# lr_pca.fit(X_train_pca, y_train)

# y_pred_pca = mahalanobis_classifier(X_train_pca, X_test_pca, y_train)
# accuracy_pca = accuracy_score(y_test, y_pred_pca)
# pca_time = time.time() - start_time

# y_pred_lda = mahalanobis_classifier(X_train_lda, X_test_lda, y_train)
# accuracy_lda = accuracy_score(y_test, y_pred_lda)
# lda_time = time.time() - start_time

# # Print results
# print("Original Accuracy: {:.3f}%".format(original_accuracy*100))
# print(f"PCA Accuracy: {accuracy_pca:.3f}")
# print(f"LDA Accuracy: {accuracy_lda:.3f}")

# print("Original time: {:.2f}s".format(original_time))
# print("PCA time: {:.2f}s".format(pca_time))
# print("LDA time: {:.2f}s".format(lda_time))

# # Visualize the comparison of PCA and LDA results
# labels = ['PCA', 'LDA']
# accuracies = [accuracy_pca, accuracy_lda]

# plt.bar(labels, accuracies)
# plt.xlabel('Dimensionality Reduction Techniques')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Comparison of PCA and LDA')
# plt.show()
