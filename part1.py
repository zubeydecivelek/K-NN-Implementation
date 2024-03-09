import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

# Load the dataset
data = pd.read_csv("English Dataset.csv")

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regular expressions
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = text.split()
    return ' '.join(tokens)

# Shuffle data
def shuffle_data(data):
    if isinstance(data, np.ndarray):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        shuffled_data = data[indices]
    elif isinstance(data, pd.DataFrame):
        shuffled_data = data.sample(frac=1, random_state=42)  # Shuffle using Pandas
    else:
        raise ValueError("Unsupported data type.")
    
    return shuffled_data

data['Text'] = data['Text'].apply(preprocess_text)

# Encode the categorical labels into numerical values
category_mapping = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}

data_encoded = data.copy()
data_encoded['Category'] = data_encoded['Category'].map(category_mapping)

# Display the first 5 rows of the preprocessed dataset
print(data_encoded.head())

def euclidean_distance(x1, x2):
    x1_dense = x1.toarray() if hasattr(x1, 'toarray') else x1
    x2_dense = x2.toarray() if hasattr(x2, 'toarray') else x2
    return np.sqrt(np.sum((x1_dense - x2_dense)**2))


def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        predicted_label = np.bincount(k_nearest_labels).argmax()
        predictions.append(predicted_label)
    return np.array(predictions)

def weighted_knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        
        # Calculate weights based on inverse distances with epsilon
        epsilon = 1e-5
        weights = 1 / (np.array(distances)[k_nearest_indices] + epsilon)
        
        # Weighted voting
        weighted_counts = np.bincount(k_nearest_labels, weights)
        predicted_label = weighted_counts.argmax()
        predictions.append(predicted_label)
    
    return np.array(predictions)

# Define functions to compute accuracy, precision, and recall for multi-class classification
def accuracy_function(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

def precision_function(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    all_positives = sum(1 for pred in y_pred if pred == 1)
    return true_positives / all_positives if all_positives != 0 else 0

def recall_function(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    actual_positives = sum(1 for true in y_true if true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0


# Create a TF-IDF vectorizer with stopwords
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, use_idf=True, lowercase=True, stop_words='english', norm='l2', min_df=1, binary=False)

# Fit on the preprocessed text data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text'])

# Get the TF-IDF feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a DataFrame to store the TF-IDF scores
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Define the categories
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

# Iterate through each category
for category in categories:
    # Filter articles belonging to the current category
    category_data = data[data['Category'] == category]
    
    # Calculate the mean TF-IDF scores for each word in the category
    category_tfidf_scores = tfidf_df.loc[category_data.index].mean()
    
    # Calculate the mean TF-IDF scores for each word in non-category articles
    non_category_tfidf_scores = tfidf_df.loc[~data.index.isin(category_data.index)].mean()
    
    # List the top 10 words whose presence predicts the category
    top_words_presence = category_tfidf_scores.sort_values(ascending=False).head(10).index
    
    # List the top 10 words whose absence predicts the category
    top_words_absence = non_category_tfidf_scores.sort_values(ascending=False).head(10).index
    
    print(f"Category: {category}")
    print("Words whose presence strongly predicts the category:\n",top_words_presence.to_list())
    print("Words whose absence strongly predicts the category:\n", top_words_absence.to_list())
    print("\n")


#------------------KNN TESTS------------------#

# Split the data into features and labels
X = data_encoded['Text'].values
y = data_encoded['Category'].values

# Define the values of k to experiment with
k_values = [1, 3]

# Initialize variables to store metrics
average_metrics_unigram = {'accuracy': {k: 0 for k in k_values},
                           'precision': {k: 0 for k in k_values},
                           'recall': {k: 0 for k in k_values}}

average_metrics_bigram = {'accuracy': {k: 0 for k in k_values},
                          'precision': {k: 0 for k in k_values},
                          'recall': {k: 0 for k in k_values}}

average_metrics_tfidf = {'accuracy': {k: 0 for k in k_values},
                         'precision': {k: 0 for k in k_values},
                         'recall': {k: 0 for k in k_values}}

# Number of folds
num_folds = 5

# 5-fold cross-validation
fold_size = len(X) // num_folds

for k in k_values:
    for fold in range(num_folds):
        # Split the data into training and testing sets for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_test_fold = X[start:end]
        y_test_fold = y[start:end]

        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = np.concatenate([y[:start], y[end:]])

        # Convert text data to numerical features using unigram BoW
        vectorizer_unigram = CountVectorizer()
        X_train_unigram = vectorizer_unigram.fit_transform(X_train_fold)
        X_test_unigram = vectorizer_unigram.transform(X_test_fold)

        # Convert text data to numerical features using bigram BoW
        vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
        X_train_bigram = vectorizer_bigram.fit_transform(X_train_fold)
        X_test_bigram = vectorizer_bigram.transform(X_test_fold)

        # Convert text data to numerical features using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, use_idf=True,
                                           lowercase=True, norm='l2', min_df=1, binary=False)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_fold)

        # Apply the unigram k-NN algorithm
        predictions_unigram = knn(X_train_unigram, y_train_fold, X_test_unigram, k)

        # Apply the bigram k-NN algorithm
        predictions_bigram = knn(X_train_bigram, y_train_fold, X_test_bigram, k)

        # Apply the TF-IDF k-NN algorithm
        predictions_tfidf = knn(X_train_tfidf, y_train_fold, X_test_tfidf, k)

        # Evaluate metrics for unigram k-NN
        accuracy_unigram = accuracy_function(predictions_unigram, y_test_fold)
        precision_unigram = precision_function(predictions_unigram, y_test_fold)
        recall_unigram = recall_function(predictions_unigram, y_test_fold)

        # Evaluate metrics for bigram k-NN
        accuracy_bigram = accuracy_function(predictions_bigram, y_test_fold)
        precision_bigram = precision_function(predictions_bigram, y_test_fold)
        recall_bigram = recall_function(predictions_bigram, y_test_fold)

        # Evaluate metrics for TF-IDF k-NN
        accuracy_tfidf = accuracy_function(predictions_tfidf, y_test_fold)
        precision_tfidf = precision_function(predictions_tfidf, y_test_fold)
        recall_tfidf = recall_function(predictions_tfidf, y_test_fold)

        # Accumulate metrics for averaging
        average_metrics_unigram['accuracy'][k] += accuracy_unigram
        average_metrics_unigram['precision'][k] += precision_unigram
        average_metrics_unigram['recall'][k] += recall_unigram

        average_metrics_bigram['accuracy'][k] += accuracy_bigram
        average_metrics_bigram['precision'][k] += precision_bigram
        average_metrics_bigram['recall'][k] += recall_bigram

        average_metrics_tfidf['accuracy'][k] += accuracy_tfidf
        average_metrics_tfidf['precision'][k] += precision_tfidf
        average_metrics_tfidf['recall'][k] += recall_tfidf


# Print the average metrics over the 5 folds for each k for unigram
print("\nResults for Unigram:")
for k in k_values:
    average_accuracy_unigram = average_metrics_unigram['accuracy'][k] / num_folds
    average_precision_unigram = average_metrics_unigram['precision'][k] / num_folds
    average_recall_unigram = average_metrics_unigram['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_unigram))
    print("Average Precision: {:.2%}".format(average_precision_unigram))
    print("Average Recall: {:.2%}".format(average_recall_unigram))

# Print the average metrics over the 5 folds for each k for bigram
print("\nResults for Bigram:")
for k in k_values:
    average_accuracy_bigram = average_metrics_bigram['accuracy'][k] / num_folds
    average_precision_bigram = average_metrics_bigram['precision'][k] / num_folds
    average_recall_bigram = average_metrics_bigram['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_bigram))
    print("Average Precision: {:.2%}".format(average_precision_bigram))
    print("Average Recall: {:.2%}".format(average_recall_bigram))

# Print the average metrics over the 5 folds for each k for TF-IDF
print("\nResults for TF-IDF:")
for k in k_values:
    average_accuracy_tfidf = average_metrics_tfidf['accuracy'][k] / num_folds
    average_precision_tfidf = average_metrics_tfidf['precision'][k] / num_folds
    average_recall_tfidf = average_metrics_tfidf['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_tfidf))
    print("Average Precision: {:.2%}".format(average_precision_tfidf))
    print("Average Recall: {:.2%}".format(average_recall_tfidf))


#------------------KNN TESTS WITH STOPWORDS------------------#

# Initialize variables to store metrics
average_metrics_unigram_stopwords = {'accuracy': {k: 0 for k in k_values},
                           'precision': {k: 0 for k in k_values},
                           'recall': {k: 0 for k in k_values}}

average_metrics_bigram_stopwords = {'accuracy': {k: 0 for k in k_values},
                          'precision': {k: 0 for k in k_values},
                          'recall': {k: 0 for k in k_values}}

average_metrics_tfidf_stopwords = {'accuracy': {k: 0 for k in k_values},
                         'precision': {k: 0 for k in k_values},
                         'recall': {k: 0 for k in k_values}}

for k in k_values:
    for fold in range(num_folds):
        # Split the data into training and testing sets for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_test_fold = X[start:end]
        y_test_fold = y[start:end]

        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = np.concatenate([y[:start], y[end:]])

        # Convert text data to numerical features using unigram BoW
        vectorizer_unigram_stopwords = CountVectorizer(stop_words='english')
        X_train_unigram_stopwords = vectorizer_unigram_stopwords.fit_transform(X_train_fold)
        X_test_unigram_stopwords = vectorizer_unigram_stopwords.transform(X_test_fold)

        # Convert text data to numerical features using bigram BoW
        vectorizer_bigram_stopwords = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        X_train_bigram_stopwords = vectorizer_bigram_stopwords.fit_transform(X_train_fold)
        X_test_bigram_stopwords = vectorizer_bigram_stopwords.transform(X_test_fold)

        # Convert text data to numerical features using TF-IDF
        tfidf_vectorizer_stopwords = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, use_idf=True,
                                           lowercase=True, norm='l2', min_df=1, binary=False, stop_words='english')
        X_train_tfidf_stopwords = tfidf_vectorizer_stopwords.fit_transform(X_train_fold)
        X_test_tfidf_stopwords = tfidf_vectorizer_stopwords.transform(X_test_fold)

        # Apply the unigram k-NN algorithm
        predictions_unigram_stopwords = knn(X_train_unigram_stopwords, y_train_fold, X_test_unigram_stopwords, k)

        # Apply the bigram k-NN algorithm
        predictions_bigram_stopwords = knn(X_train_bigram_stopwords, y_train_fold, X_test_bigram_stopwords, k)

        # Apply the TF-IDF k-NN algorithm
        predictions_tfidf_stopwords = knn(X_train_tfidf_stopwords, y_train_fold, X_test_tfidf_stopwords, k)

        # Evaluate metrics for unigram k-NN
        accuracy_unigram_stopwords = accuracy_function(predictions_unigram_stopwords, y_test_fold)
        precision_unigram_stopwords = precision_function(predictions_unigram_stopwords, y_test_fold)
        recall_unigram_stopwords = recall_function(predictions_unigram_stopwords, y_test_fold)

        # Evaluate metrics for bigram k-NN
        accuracy_bigram_stopwords = accuracy_function(predictions_bigram_stopwords, y_test_fold)
        precision_bigram_stopwords = precision_function(predictions_bigram_stopwords, y_test_fold)
        recall_bigram_stopwords = recall_function(predictions_bigram_stopwords, y_test_fold)

        # Evaluate metrics for TF-IDF k-NN
        accuracy_tfidf_stopwords = accuracy_function(predictions_tfidf_stopwords, y_test_fold)
        precision_tfidf_stopwords = precision_function(predictions_tfidf_stopwords, y_test_fold)
        recall_tfidf_stopwords = recall_function(predictions_tfidf_stopwords, y_test_fold)

        # Accumulate metrics for averaging
        average_metrics_unigram_stopwords['accuracy'][k] += accuracy_unigram_stopwords
        average_metrics_unigram_stopwords['precision'][k] += precision_unigram_stopwords
        average_metrics_unigram_stopwords['recall'][k] += recall_unigram_stopwords

        average_metrics_bigram_stopwords['accuracy'][k] += accuracy_bigram_stopwords
        average_metrics_bigram_stopwords['precision'][k] += precision_bigram_stopwords
        average_metrics_bigram_stopwords['recall'][k] += recall_bigram_stopwords

        average_metrics_tfidf_stopwords['accuracy'][k] += accuracy_tfidf_stopwords
        average_metrics_tfidf_stopwords['precision'][k] += precision_tfidf_stopwords
        average_metrics_tfidf_stopwords['recall'][k] += recall_tfidf_stopwords

# Print the average metrics over the 5 folds for each k for unigram
print("\nResults for Unigram with stopwords:")
for k in k_values:
    average_accuracy_unigram = average_metrics_unigram_stopwords['accuracy'][k] / num_folds
    average_precision_unigram = average_metrics_unigram_stopwords['precision'][k] / num_folds
    average_recall_unigram = average_metrics_unigram_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_unigram))
    print("Average Precision: {:.2%}".format(average_precision_unigram))
    print("Average Recall: {:.2%}".format(average_recall_unigram))

# Print the average metrics over the 5 folds for each k for bigram
print("\nResults for Bigram with stopwords:")
for k in k_values:
    average_accuracy_bigram = average_metrics_bigram_stopwords['accuracy'][k] / num_folds
    average_precision_bigram = average_metrics_bigram_stopwords['precision'][k] / num_folds
    average_recall_bigram = average_metrics_bigram_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_bigram))
    print("Average Precision: {:.2%}".format(average_precision_bigram))
    print("Average Recall: {:.2%}".format(average_recall_bigram))

# Print the average metrics over the 5 folds for each k for TF-IDF
print("\nResults for TF-IDF with stopwords:")
for k in k_values:
    average_accuracy_tfidf = average_metrics_tfidf_stopwords['accuracy'][k] / num_folds
    average_precision_tfidf = average_metrics_tfidf_stopwords['precision'][k] / num_folds
    average_recall_tfidf = average_metrics_tfidf_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_tfidf))
    print("Average Precision: {:.2%}".format(average_precision_tfidf))
    print("Average Recall: {:.2%}".format(average_recall_tfidf))



#------------------WEIGHTED KNN TESTS------------------#

# Initialize variables to store metrics
average_metrics_unigram_weighted = {'accuracy': {k: 0 for k in k_values},
                                     'precision': {k: 0 for k in k_values},
                                     'recall': {k: 0 for k in k_values}}

average_metrics_bigram_weighted = {'accuracy': {k: 0 for k in k_values},
                                    'precision': {k: 0 for k in k_values},
                                    'recall': {k: 0 for k in k_values}}

average_metrics_tfidf_weighted = {'accuracy': {k: 0 for k in k_values},
                                   'precision': {k: 0 for k in k_values},
                                   'recall': {k: 0 for k in k_values}}

for k in k_values:
    for fold in range(num_folds):
        # Split the data into training and testing sets for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_test_fold = X[start:end]
        y_test_fold = y[start:end]

        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = np.concatenate([y[:start], y[end:]])

        # Convert text data to numerical features using unigram BoW 
        vectorizer_unigram = CountVectorizer()
        X_train_unigram = vectorizer_unigram.fit_transform(X_train_fold)
        X_test_unigram = vectorizer_unigram.transform(X_test_fold)

        # Convert text data to numerical features using bigram BoW with stopwords
        vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
        X_train_bigram = vectorizer_bigram.fit_transform(X_train_fold)
        X_test_bigram = vectorizer_bigram.transform(X_test_fold)

        # Convert text data to numerical features using TF-IDF with stopwords
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, use_idf=True,
                                           lowercase=True, norm='l2', min_df=1, binary=False)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_fold)

        # Apply the unigram weighted k-NN algorithm
        predictions_unigram_weighted = weighted_knn(X_train_unigram, y_train_fold, X_test_unigram, k)

        # Apply the bigram weighted k-NN algorithm
        predictions_bigram_weighted = weighted_knn(X_train_bigram, y_train_fold, X_test_bigram, k)

        # Apply the TF-IDF weighted k-NN algorithm
        predictions_tfidf_weighted = weighted_knn(X_train_tfidf, y_train_fold, X_test_tfidf, k)

        # Evaluate metrics for weighted unigram k-NN
        accuracy_unigram_weighted = accuracy_function(predictions_unigram_weighted, y_test_fold)
        precision_unigram_weighted = precision_function(predictions_unigram_weighted, y_test_fold)
        recall_unigram_weighted = recall_function(predictions_unigram_weighted, y_test_fold)

        # Evaluate metrics for weighted bigram k-NN
        accuracy_bigram_weighted = accuracy_function(predictions_bigram_weighted, y_test_fold)
        precision_bigram_weighted = precision_function(predictions_bigram_weighted, y_test_fold)
        recall_bigram_weighted = recall_function(predictions_bigram_weighted, y_test_fold)

        # Evaluate metrics for weighted TF-IDF k-NN
        accuracy_tfidf_weighted = accuracy_function(predictions_tfidf_weighted, y_test_fold)
        precision_tfidf_weighted = precision_function(predictions_tfidf_weighted, y_test_fold)
        recall_tfidf_weighted = recall_function(predictions_tfidf_weighted, y_test_fold)

        # Accumulate metrics for averaging
        average_metrics_unigram_weighted['accuracy'][k] += accuracy_unigram_weighted
        average_metrics_unigram_weighted['precision'][k] += precision_unigram_weighted
        average_metrics_unigram_weighted['recall'][k] += recall_unigram_weighted

        average_metrics_bigram_weighted['accuracy'][k] += accuracy_bigram_weighted
        average_metrics_bigram_weighted['precision'][k] += precision_bigram_weighted
        average_metrics_bigram_weighted['recall'][k] += recall_bigram_weighted

        average_metrics_tfidf_weighted['accuracy'][k] += accuracy_tfidf_weighted
        average_metrics_tfidf_weighted['precision'][k] += precision_tfidf_weighted
        average_metrics_tfidf_weighted['recall'][k] += recall_tfidf_weighted



# Print the average metrics over the 5 folds for each k for weighted unigram
print("\nResults for Weighted Unigram:")
for k in k_values:
    average_accuracy_unigram_weighted = average_metrics_unigram_weighted['accuracy'][k] / num_folds
    average_precision_unigram_weighted = average_metrics_unigram_weighted['precision'][k] / num_folds
    average_recall_unigram_weighted = average_metrics_unigram_weighted['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_unigram_weighted))
    print("Average Precision: {:.2%}".format(average_precision_unigram_weighted))
    print("Average Recall: {:.2%}".format(average_recall_unigram_weighted))

# Print the average metrics over the 5 folds for each k for weighted bigram
print("\nResults for Weighted Bigram:")
for k in k_values:
    average_accuracy_bigram_weighted = average_metrics_bigram_weighted['accuracy'][k] / num_folds
    average_precision_bigram_weighted = average_metrics_bigram_weighted['precision'][k] / num_folds
    average_recall_bigram_weighted = average_metrics_bigram_weighted['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_bigram_weighted))
    print("Average Precision: {:.2%}".format(average_precision_bigram_weighted))
    print("Average Recall: {:.2%}".format(average_recall_bigram_weighted))

# Print the average metrics over the 5 folds for each k for weighted TF-IDF
print("\nResults for Weighted TF-IDF:")
for k in k_values:
    average_accuracy_tfidf_weighted = average_metrics_tfidf_weighted['accuracy'][k] / num_folds
    average_precision_tfidf_weighted = average_metrics_tfidf_weighted['precision'][k] / num_folds
    average_recall_tfidf_weighted = average_metrics_tfidf_weighted['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_tfidf_weighted))
    print("Average Precision: {:.2%}".format(average_precision_tfidf_weighted))
    print("Average Recall: {:.2%}".format(average_recall_tfidf_weighted))


#------------------WEIGHTED KNN TESTS WITH STOPWORDS------------------#

# Initialize variables to store metrics
average_metrics_unigram_weighted_stopwords = {'accuracy': {k: 0 for k in k_values},
                                     'precision': {k: 0 for k in k_values},
                                     'recall': {k: 0 for k in k_values}}

average_metrics_bigram_weighted_stopwords = {'accuracy': {k: 0 for k in k_values},
                                    'precision': {k: 0 for k in k_values},
                                    'recall': {k: 0 for k in k_values}}

average_metrics_tfidf_weighted_stopwords = {'accuracy': {k: 0 for k in k_values},
                                   'precision': {k: 0 for k in k_values},
                                   'recall': {k: 0 for k in k_values}}


for k in k_values:
    for fold in range(num_folds):
        # Split the data into training and testing sets for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_test_fold = X[start:end]
        y_test_fold = y[start:end]

        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = np.concatenate([y[:start], y[end:]])

        # Convert text data to numerical features using unigram BoW with stopwords
        vectorizer_unigram = CountVectorizer(stop_words='english')
        X_train_unigram = vectorizer_unigram.fit_transform(X_train_fold)
        X_test_unigram = vectorizer_unigram.transform(X_test_fold)

        # Convert text data to numerical features using bigram BoW with stopwords
        vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        X_train_bigram = vectorizer_bigram.fit_transform(X_train_fold)
        X_test_bigram = vectorizer_bigram.transform(X_test_fold)

        # Convert text data to numerical features using TF-IDF with stopwords
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, use_idf=True,
                                           lowercase=True, norm='l2', min_df=1, binary=False, stop_words='english')
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_fold)

        # Apply the unigram weighted k-NN algorithm
        predictions_unigram_weighted = weighted_knn(X_train_unigram, y_train_fold, X_test_unigram, k)

        # Apply the bigram weighted k-NN algorithm
        predictions_bigram_weighted = weighted_knn(X_train_bigram, y_train_fold, X_test_bigram, k)

        # Apply the TF-IDF weighted k-NN algorithm
        predictions_tfidf_weighted = weighted_knn(X_train_tfidf, y_train_fold, X_test_tfidf, k)

        # Evaluate metrics for weighted unigram k-NN
        accuracy_unigram_weighted = accuracy_function(predictions_unigram_weighted, y_test_fold)
        precision_unigram_weighted = precision_function(predictions_unigram_weighted, y_test_fold)
        recall_unigram_weighted = recall_function(predictions_unigram_weighted, y_test_fold)

        # Evaluate metrics for weighted bigram k-NN
        accuracy_bigram_weighted = accuracy_function(predictions_bigram_weighted, y_test_fold)
        precision_bigram_weighted = precision_function(predictions_bigram_weighted, y_test_fold)
        recall_bigram_weighted = recall_function(predictions_bigram_weighted, y_test_fold)

        # Evaluate metrics for weighted TF-IDF k-NN
        accuracy_tfidf_weighted = accuracy_function(predictions_tfidf_weighted, y_test_fold)
        precision_tfidf_weighted = precision_function(predictions_tfidf_weighted, y_test_fold)
        recall_tfidf_weighted = recall_function(predictions_tfidf_weighted, y_test_fold)

        # Accumulate metrics for averaging
        average_metrics_unigram_weighted_stopwords['accuracy'][k] += accuracy_unigram_weighted
        average_metrics_unigram_weighted_stopwords['precision'][k] += precision_unigram_weighted
        average_metrics_unigram_weighted_stopwords['recall'][k] += recall_unigram_weighted

        average_metrics_bigram_weighted_stopwords['accuracy'][k] += accuracy_bigram_weighted
        average_metrics_bigram_weighted_stopwords['precision'][k] += precision_bigram_weighted
        average_metrics_bigram_weighted_stopwords['recall'][k] += recall_bigram_weighted

        average_metrics_tfidf_weighted_stopwords['accuracy'][k] += accuracy_tfidf_weighted
        average_metrics_tfidf_weighted_stopwords['precision'][k] += precision_tfidf_weighted
        average_metrics_tfidf_weighted_stopwords['recall'][k] += recall_tfidf_weighted


# Print the average metrics over the 5 folds for each k for weighted unigram
print("\nResults for Weighted Unigram with stopwords:")
for k in k_values:
    average_accuracy_unigram_weighted_stopwords = average_metrics_unigram_weighted_stopwords['accuracy'][k] / num_folds
    average_precision_unigram_weighted_stopwords = average_metrics_unigram_weighted_stopwords['precision'][k] / num_folds
    average_recall_unigram_weighted_stopwords = average_metrics_unigram_weighted_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_unigram_weighted_stopwords))
    print("Average Precision: {:.2%}".format(average_precision_unigram_weighted_stopwords))
    print("Average Recall: {:.2%}".format(average_recall_unigram_weighted_stopwords))

# Print the average metrics over the 5 folds for each k for weighted bigram
print("\nResults for Weighted Bigram with stopwords:")
for k in k_values:
    average_accuracy_bigram_weighted_stopwords = average_metrics_bigram_weighted_stopwords['accuracy'][k] / num_folds
    average_precision_bigram_weighted_stopwords = average_metrics_bigram_weighted_stopwords['precision'][k] / num_folds
    average_recall_bigram_weighted_stopwords = average_metrics_bigram_weighted_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_bigram_weighted_stopwords))
    print("Average Precision: {:.2%}".format(average_precision_bigram_weighted_stopwords))
    print("Average Recall: {:.2%}".format(average_recall_bigram_weighted_stopwords))

# Print the average metrics over the 5 folds for each k for weighted TF-IDF
print("\nResults for Weighted TF-IDF with stopwords:")
for k in k_values:
    average_accuracy_tfidf_weighted_stopwords = average_metrics_tfidf_weighted_stopwords['accuracy'][k] / num_folds
    average_precision_tfidf_weighted_stopwords = average_metrics_tfidf_weighted_stopwords['precision'][k] / num_folds
    average_recall_tfidf_weighted_stopwords = average_metrics_tfidf_weighted_stopwords['recall'][k] / num_folds

    print(f"\nResults for k = {k}:")
    print("Average Accuracy: {:.2%}".format(average_accuracy_tfidf_weighted_stopwords))
    print("Average Precision: {:.2%}".format(average_precision_tfidf_weighted_stopwords))
    print("Average Recall: {:.2%}".format(average_recall_tfidf_weighted_stopwords))

