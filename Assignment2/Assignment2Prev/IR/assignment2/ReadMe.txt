Assignment 2:

Overview
This application performs text classification using three different classifiers: Multinomial Naive Bayes, k-Nearest Neighbors, and Support Vector Machine. It is designed to classify documents into two categories: 'Windows' and 'Hockey'. 

Code Structure:
- load_data(): Loads the dataset from the provided files.
- train_multinomial_nb(): Trains the Multinomial Naive Bayes classifier.
- apply_multinomial_nb(): Applies the trained classifier to test data.
- evaluate_classifier(): Evaluates the classifier's performance using various metrics.
- save_confusion_matrix(): Saves the confusion matrix as an image file.
- plot_evaluation_metrics(): Plots and saves evaluation metrics as line graphs.
- save_evaluation_metrics(): Saves evaluation metrics to a text file.
- main(): Main function to execute the entire pipeline.

## Launching the Application:

To launch the application, follow these steps:
1. Ensure that Python 3.x is installed on your system. You can download Python from the official website: python.org.

2. Create and Activate a Virtual Environment
   - pip install virtualenv
   - virtualenv env

   # Activate the virtual environment
   # On Windows
   - env\Scripts\activate
   # On macOS/Linux
   - source env/bin/activate

3. Install Required Dependencies
   - pip install -r requirements.txt

4. Run the Main File.
   # On Windows
   - python assignment2.py
   # On macOS/Linux
   - python3 assignment2.py

5. Output files will be generated in the 'same' directory.

External Libraries Used
- NumPy: For numerical computations.
- scikit-learn: For machine learning algorithms and evaluation metrics.
- Matplotlib: For visualization of confusion matrices and evaluation metrics.

Additional Information
- The code assumes two classes: 'Windows' and 'Hockey'.
- Modify the file paths in the code if the dataset location changes.
- Experiment with different parameters for k-Nearest Neighbors (k) and SVM if needed.

TEAM MEMBERS:
  NAME: Balraj Hanmanthugari        UID: U01079536               EMAIL: hanmanthugari.2@wright.edu
  NAME: Deepika Kasula              UID: U01067608               EMAIL: kasula.16@wright.edu
  NAME: Nitish Kota                 UID: U01074656               EMAIL: kota.58@wright.edu