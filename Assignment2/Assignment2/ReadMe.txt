Team
Mourya Gottipati    -    U01099603    -    gottipati.22@wright.edu
Niranjan Kumar Ramineni    -    U01065247    -    ramineni.15@wright.edu

Python version - 3.10.11

Launching Application
1.) open terminal
2.) Navigate to working directory
3.) run command  -  "python assignment2.py"

After running this file the resultant graphs and metrics text documents will be stored in the working directory

Libraries used
NumPy: Used for numerical operations and handling arrays and matrices.
scikit-learn: Used for machine learning models and evaluation metrics.
matplotlib: Used for plotting and visualizing data.

Application:

This application implements and evaluates a Bernoulli Naive Bayes (BNB) text classifier using two approaches those are a custom implementation and the scikit-learn library's implementation.

1. Data Loading:
    - The term-document matrices for training and testing data are loaded from files (trainMatrixModified.txt and testMatrixModified.txt).
    - The class labels for the training and testing datasets are also loaded from files(trainClasses.txt and testClasses.txt).

2. Manual Bernoulli Naive Bayes Classifier:
    Training: The trainBernoulliNB function calculates the prior probabilities for each class and the conditional probabilities of each term given the class.
    Prediction: The applyBernoulliNB function uses the trained model to classify new documents by calculating the log-probability scores for each class and selecting the class with the highest score.

3. Scikit-learn Bernoulli Naive Bayes Classifier:
    - The BernoulliNB class from scikit-learn is used to train a classifier on the training data.
    - The trained classifier is then used to predict the class labels for the test data.

4. Evaluation and Metrics:
    - The writeMetrics function computes and writes evaluation metrics (accuracy, precision, recall, F1-score, and confusion matrix) for both classifiers to text files.
    - It also generates and saves plots of the evaluation metrics and the confusion matrices for visual analysis.

This application provides a comprehensive comparison of a manually implemented BNB classifier with a standard library implementation, showcasing the effectiveness and correctness of both methods.