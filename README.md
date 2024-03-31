# Website-Phishing Classification

The "Website-Phishing Classification" project aims to develop a machine learning model to classify websites as either legitimate or phishing based on various features extracted from the website's characteristics. The project utilizes a dataset containing labeled examples of legitimate and phishing websites, and employs machine learning techniques to build a predictive model.
Used Support Vector Classifier, Decision Tree Classifier, Random Forest Classifier, Logistic Regression and computed the accuracy, f1-score and classification report.

[Dataset taken from Kaggle](https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset/data)

**Features**<br>
*   url_length: The length of the URL
*   n_dots: The count of ‘.’ characters in the URL.
*   n_hypens: The count of ‘-’ characters in the URL.
*   n_underline: The count of ‘_’ characters in the URL.
*   n_slash: The count of ‘/’ characters in the URL.
*   n_dots: The count of ‘.’ characters in the URL.
*   n_questionmark: The count of ‘?’ characters in the URL.
*   n_equal: The count of ‘=’ characters in the URL.
*   n_at: The count of ‘@’ characters in the URL.
*   n_and: The count of ‘&’ characters in the URL.
*   n_exclamation: The count of ‘!’ characters in the URL.
*   n_space: The count of ’ ’ characters in the URL
*   n_tilde: The count of ‘~’ characters in the URL
*   n_comma: The count of ‘,’ characters in the URL.
*   n_plus: The count of ‘+’ characters in the URL
*   n_asterisk: The count of ‘*’ characters in the URL.
*   n_hastag: The count of ‘#’ characters in the URL.
*   n_dollar: The count of ‘$’ characters in the URL.
*   n_percent: The count of ‘%’ characters in the URL.
*  n_redirection: The count of redirections in the URL.
*   phishing: The Labels of the URL. 1 is phishing and 0 is legitimate.
##

Random Forest Classifier gives an impressive accuracy of 86% on unseen data and no underfitting or overfitting conditions.


## Libraries Used
Used Python as programming language
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (sklearn)

## Procedure
#### 1. Data Visualization
    Displaying dataframe, correlation matrix, class distribution
#### 2. Pre-processing
    Checking null/missing values, removing duplicates, balancing class distribution
    Standardize the features, upsampling distribution
#### 3. Model Selection and Hyperparamter Tuning (Used GridSearchCV to get best parameters)
    GridSearchCV, Logistic Regression, Decision Tree Classifier, Random Forest Classifier (Ensemble Technique)
    Support Vector classifier
#### 4. Training the model
    Training the model with best parameters from hyperparamter tuning
#### 5. Predicting and evaluating results
      Computing predictions, accuracy-score, f1-score, precison and classification resutls
#### 6. Visualizing the results
      Computing Confusion Matrix to get True,False Positives and True,False Negatives


## Results
### Decision Tree Classifier

    Train Accuracy : 94.72
    Test Accuracy: 85.87

Classicication Report


                  precision    recall  f1-score   support

             0       0.82      0.95      0.88      3230
             1       0.94      0.78      0.85      3119

      accuracy                           0.86      6349
     macro_avg       0.88      0.86      0.86      6349
    weight_avg       0.87      0.86      0.86      6349


### Random Forest Classifier

    Train Accuracy : 94.72
    Test Accuracy: 86.28
    
Classicication Report

                 precision    recall  f1-score   support

             0       0.83      0.92      0.87      3230
             1       0.91      0.80      0.85      3119

      accuracy                           0.86      6349
     macro avg       0.87      0.86      0.86      6349
    weight_avg       0.87      0.86      0.86      6349

Results from Logsitic Regression and SVC are in notebook. Models with best accuracy scores are listed above.

##



