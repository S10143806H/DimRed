# Install dependences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def mahalanobis_classifier(X_train, y_train, X_test, y_test):
    unique_labels = np.unique(y_train)
    cov_matrices = {}
    means = {}
    cov_inv = {}

    # Calculate the covariance matrices and means for each class
    for label in unique_labels:
        data = X_train[y_train == label]
        cov_matrices[label] = np.cov(data.T)
        means[label] = np.mean(data, axis=0)
        cov_inv[label] = np.linalg.pinv(cov_matrices[label])

    # Perform classification using minimum Mahalanobis distance
    y_pred = []
    for x in X_test:
        distances = []
        for label in unique_labels:
            distance = mahalanobis(x, means[label], cov_inv[label])
            distances.append(distance)
        y_pred.append(unique_labels[np.argmin(distances)])

    return np.array(y_pred)


# Pre processing
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)

# Normalization
X = lfw_people.data
y = lfw_people.target
n_classes = lfw_people.target_names.shape[0]
component = 30


print(X.shape)
print(y.shape)

print("Number of features:", X.shape[1])            # 5655
print("Number of datasets (samples):", y.shape[0])  # 4061
print("Number of classes: ", n_classes)

# Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Apply PCA
error_rates_pca = []
error_rates_pca_model = []
for n_components_pca in range(component, 1, -1):
    pca = PCA(n_components=n_components_pca, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    y_pred_pca = mahalanobis_classifier(
        X_train_pca, y_train, X_test_pca, y_test)
    error_rate_pca = 1 - np.sum(y_pred_pca == y_test) / len(y_test)
    error_rates_pca.append(error_rate_pca)
    print(
        f"PCA Error Rate with {n_components_pca} components:", error_rate_pca)

    # Train and evaluate model on PCA-reduced data
    clf_pca = SVC(kernel='linear', C=1)
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    error_rate_pca_model = 1 - accuracy_pca
    error_rates_pca_model.append(error_rate_pca_model)
    print(
        f"PCA Model Error Rate with {n_components_pca} components:", error_rate_pca_model)


# Apply LDA
error_rates_lda = []
error_rates_lda_model = []
for n_components_lda in range(component, 1, -1):
    lda = LDA(n_components=n_components_lda)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    y_pred_lda = mahalanobis_classifier(
        X_train_lda, y_train, X_test_lda, y_test)
    error_rate_lda = 1 - np.sum(y_pred_lda == y_test) / len(y_test)
    error_rates_lda.append(error_rate_lda)
    print(
        f"LDA Error Rate with {n_components_lda} components:", error_rate_lda)

    # Train and evaluate model on LDA-reduced data
    clf_lda = SVC(kernel='linear', C=1)
    clf_lda.fit(X_train_lda, y_train)
    y_pred_lda = clf_lda.predict(X_test_lda)
    accuracy_lda = accuracy_score(y_test, y_pred_lda)
    error_rate_lda_model = 1 - accuracy_lda
    error_rates_lda_model.append(error_rate_lda_model)
    print(
        f"LDA Model Error Rate with {n_components_lda} components:", error_rate_lda_model)

# Plot the accuracies
plt.plot(range(component, 1, -1), error_rates_pca, marker='x', label='PCA')
plt.plot(range(component, 1, -1), error_rates_pca_model,
         marker='x', label='PCA_Model')
plt.plot(range(component, 1, -1), error_rates_lda, marker='o', label='LDA')
plt.plot(range(component, 1, -1), error_rates_lda_model,
         marker='o', label='LDA_Model')
plt.xlabel('Number of Components (Descending)')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Number of Components')
plt.legend()
plt.grid()
plt.gca().invert_xaxis()
plt.show()
