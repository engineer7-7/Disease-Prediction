Disease Prediction
The Disease Prediction project is a machine learning application that predicts the likelihood of a disease based on provided symptoms and features. Various machine learning models, such as K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM), are used for training and evaluating model accuracy.

Files
Training.csv: The training dataset containing data used for model training.
Testing.csv: The testing dataset used for evaluating the model's performance (though not used directly in this project).
Key Libraries
This project uses the following libraries:

pandas and numpy for data manipulation
matplotlib and seaborn for visualization
sklearn for machine learning models and evaluation metrics
Steps
1. Data Loading and Inspection
Load the Training.csv file and check for any missing values (null or NaN).
Drop columns with missing values.
2. Data Visualization
Visualize the distribution of the target column (prognosis) using a bar chart.
3. Data Preprocessing
Convert the target column (prognosis) from string values to numerical values using LabelEncoder.
4. Dataset Splitting
Split the data into features (X) and target (y).
Use train_test_split to divide the dataset into training and testing sets (80% - 20%).
5. ML Training and Prediction
The following models were implemented:

K-Nearest Neighbors (KNN):

Train with KNeighborsClassifier.
Make predictions using random data and convert the prediction back to its original value.
Naive Bayes (GaussianNB):

Train using GaussianNB.
Predict and display the model's accuracy.
Support Vector Machine (SVM):

Train using SVC.
Predict and display the model's accuracy.
6. Evaluation of Results
Display the accuracy and other metrics (precision, recall, F1-score) for each model using sklearn metrics.
Results and Conclusions
The accuracy and performance metrics for each model are provided as an indication of how well they predict the disease.
A comparison between KNN, Naive Bayes, and SVM shows which model is more effective for this dataset.
Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
