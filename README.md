# K-Nearest Neighbor Algorithm Implementation

## Part 1: Classification of News Articles

### Dataset
The dataset for Part 1 can be found [here](https://www.kaggle.com/datasets/qusaybtoush1990/english-news), containing 1491 samples with 5 discrete ground-truth class types: Sport, Business, Politics, Entertainment, Tech.

### Approach

#### 1.1 Understanding the Data
Implemented a nearest neighbor algorithm to predict the category of news articles using a Bag of Words (BoW) model. Feasible keywords, such as 'election,' 'technology,' and 'profit,' were analyzed for their frequency in each category.

#### 1.2 Implementing k Nearest Neighbor
- Utilized the k Nearest Neighbor algorithm to classify articles.
- Represented data using BoW with options for Unigram and Bigram.
- Handled unseen words during testing by either ignoring or assigning a non-zero default value.

#### 1.3 Analyzing the Words
1. Listed 10 words predicting the article's category presence and absence for each category.
2. Explored TF-IDF and Information Theory for word selection.

#### 1.4 Stopwords
Identified 10 non-stopwords predicting article categories for each category and discussed the relevance of removing or keeping stopwords.

### Classification Performance Metric
Computed Accuracy, Precision, and Recall for each test using 5-fold cross-validation. Reported average metrics.

### Error Analysis for Classification
- Identified misclassified samples and discussed challenges.
- Compared feature choices, system parameters (e.g., k), and computation time.

## Part 2: Medical Insurance Cost Estimation from Data

### Regression Dataset
The dataset for Part 2 can be found [here](https://www.kaggle.com/datasets/mirichoi0218/insurance), with 1338 samples and continuous medical cost values.

### Regression Performance Metric
Calculated Mean Absolute Error (MAE) for each test using 5-fold cross-validation. Reported average MAE.

### Feature Normalization
Used min-max normalization to re-scale feature columns between 0-1.

### Error Analysis for Regression
- Compared feature normalization choices.
- Investigated the effect of system parameters on performance.
- Discussed computation time in addition to regression rates.

## Implementation Details
- Implemented K-fold cross-validation, shuffle methods, k-NN, weighted k-NN, accuracy, precision, recall, MAE, and min-max feature normalization without using ready-made libraries.
- Used Numpy array functions for intermediate steps.
