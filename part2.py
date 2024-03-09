import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to your CSV file in Google Drive
csv_file_path = "/content/drive/MyDrive/BBM409_datasets/insurance.csv"

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)

df.head()

# MIN-MAX NORMALIZATION
def min_max_normalization(df, columns_to_normalize):
    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()

        if min_val == max_val:
            df[column] = 0  # Avoid division by zero
        else:
            df[column] = (df[column] - min_val) / (max_val - min_val)

#Euclidean Distance
def get_euclidian_distance(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

# Manhattan Distance
def get_manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# mae function
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

"""# Pre-Processing"""

df_target = df['charges']
df_var = df.iloc[:, :-1]

# min-max normalization of numeric data
min_max_normalization(df_var, ['age', 'bmi', 'children'])

print(df_var.head(10))

# one-hot encoding sex, region and smoker
# they will be one-hot encoded, because there is no hierarchical relationship between them
df_var = pd.get_dummies(df_var, columns=['sex'], prefix=['sex'])
df_var = pd.get_dummies(df_var, columns=['smoker'], prefix=['smoker'])
df_var = pd.get_dummies(df_var, columns=['region'], prefix=['region'])

df_ = pd.concat([df_var, df_target], axis=1)

df_.head()

"""# KNN"""

def KNN(X_train, X_test, y_train, k_val):
    predictions = []

    for test_data in X_test:
        distances = []
        for train_data in X_train:
            distances.append(get_euclidian_distance(train_data, test_data))

        k_neighbors_indices = np.argsort(distances)[:k_val]
        k_neighbor_labels = y_train[k_neighbors_indices]

        # Find the most common label among the k neighbors
        most_common_label = Counter(k_neighbor_labels).most_common(1)[0][0]

        predictions.append(most_common_label)

    return np.array(predictions)

data = df_.values

# split data into features (X) and target (y)
X = data[:, :-1]
y = data[:, -1]

num_folds = 5

mae_values = []
mae_dict= {}

fold_size = len(X) // num_folds

# k-NN with different values of k for each fold
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    k_mae_values = []
    for i in range(num_folds):
        # split the data into training and validation sets for this fold
        start = i * fold_size
        end = (i + 1) * fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        y_pred = KNN(X_train, X_val, y_train, k)
        mae = calculate_mae(y_val, y_pred)
        mae_values.append(mae)
        k_mae_values.append(mae)
        print(f"mae in fold {i+1} of k-value {k} : {mae}")

    #calculate the average MAE of the current fold
    mean_mae = np.mean(k_mae_values)
    mae_dict[k] = mean_mae

# calculate the average MAE across all folds and k values
average_mae = np.mean(mae_values)
print(f'Average MAE: {average_mae:.2f}')

"""# Weighted kNN"""

def weighted_KNN(X_train, X_test, y_train, k_val):
    predictions = []

    for test_data in X_test:
        distances = []
        for train_data in X_train:
            distances.append(get_euclidian_distance(train_data, test_data))

        k_neighbors_indices = np.argsort(distances)[:k_val]
        k_neighbor_labels = y_train[k_neighbors_indices]
        k_neighbor_distances = np.array(distances)[k_neighbors_indices]

        weights = 1 / (k_neighbor_distances + 1e-6)  # Adding a small value to avoid division by zero

        weighted_sum = np.sum(weights * k_neighbor_labels)

        weighted_prediction = int(round(weighted_sum / np.sum(weights)))

        predictions.append(weighted_prediction)

    return predictions

data = df_.values

# split data into features (X) and target (y)
X = data[:, :-1]
y = data[:, -1]

num_folds = 5

mae_values_weighted = []
mae_dict_weightes = {}

fold_size = len(X) // num_folds

# k-NN with different values of k for each fold
k_values = [1, 3, 5, 7 ,9]
for k in k_values:
    k_mae_values = []
    for i in range(num_folds):
        # split the data into training and validation sets for this fold
        start = i * fold_size
        end = (i + 1) * fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        y_pred = weighted_KNN(X_train, X_val, y_train,  k)
        mae = calculate_mae(y_val, y_pred)
        mae_values_weighted.append(mae)
        k_mae_values.append(mae)
        print(f"mae in fold {i+1} of k-value {k} : {mae}")

    # calculate the average MAE across all folds and k values
    mean_mae = np.mean(k_mae_values)
    mae_dict_weightes[k] = mean_mae

# calculate the average MAE across all folds and k values
average_mae_weighted = np.mean(mae_values_weighted)
print(f'Average MAE: {average_mae_weighted:.2f}')

"""#Error Analysis for Regression"""

k_values_1 = list(mae_dict.keys())
mean_mae_values_1 = list(mae_dict.values())
k_values_2 = list(mae_dict_weightes.keys())
mean_mae_values_2 = list(mae_dict_weightes.values())

# average MAE for each dictionary
average_mae_knn = np.mean(mean_mae_values_1)
average_mae_weighted_knn = np.mean(mean_mae_values_2)

plt.figure(figsize=(14, 6))

# subplot for mae_dict
plt.subplot(1, 2, 1)
plt.plot(k_values_1, mean_mae_values_1,marker='o', color='purple', label='Individual MAE')
plt.axhline(average_mae_knn, color='blue', linestyle='--', label='Average MAE')
plt.xlabel('k-values')
plt.ylabel('Mean MAE')
plt.title('Mean MAE vs. k-values (KNN)')
plt.xticks(k_values_1)

# subplot for mae_dict_weightes
plt.subplot(1, 2, 2)
plt.plot(k_values_2, mean_mae_values_2,marker='o', color='blue', label='Individual MAE')
plt.axhline(average_mae_weighted_knn, color='purple', linestyle='--', label='Average MAE')
plt.xlabel('k-values')
plt.ylabel('Mean MAE')
plt.title('Mean MAE vs. k-values (Weighted KNN)')
plt.xticks(k_values_2)

plt.tight_layout()
plt.show()

"""*   For "KNN", across the different values of k, there is a consistent MAE value with a slight difference in k value 9. this shows that, choice of k value doesn't have a significant impact on the MAE.

*   For "Weighted KNN", thw MAE values differantiate across the varios k values. The best MAE is in k=3. For the rest of the values, as k increase, error value increases, too.

*   Overall results are as follows;
    1. KNN Average MAE: 3545.23
    2. Weighted KNN Average MAE: 3480.48

*   Weighted KNN has a better performance in terms of average MAE values.

Weighted KNN has better performance than KNN due to its ability to assign variable weights to neighbour points, capturing local patterns. The coice of 'k' can influence the results.




"""



