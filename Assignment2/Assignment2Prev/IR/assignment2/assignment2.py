import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# classes
classes = ['Windows', 'Hockey']


# Load data
def load_data():
    train_matrix_data = np.loadtxt('2Newsgroups/trainMatrixModified.txt')
    test_matrix_data = np.loadtxt('2Newsgroups/testMatrixModified.txt')

    train_classes = np.loadtxt('2Newsgroups/trainClasses.txt', dtype=int)[:, 1]
    test_classes = np.loadtxt('2Newsgroups/testClasses.txt', dtype=int)[:, 1]

    with open('2Newsgroups/modifiedterms.txt', 'r') as file:
        voc = file.read().splitlines()

    return train_matrix_data, test_matrix_data, train_classes, test_classes, voc


# Multinomial Naive Bayes classifier
def train_multinomial_nb(train_matrix_data, train_classes):
    N = train_matrix_data.shape[1]
    C = np.unique(train_classes)
    vocabulary_size = train_matrix_data.shape[0]

    priors = np.zeros(len(C))
    cond_probs = np.zeros((vocabulary_size, len(C)))

    for c in C:
        train_matrix_data_c = train_matrix_data[:, train_classes == c]
        Nc = train_matrix_data_c.shape[1]

        priors[c] = Nc / N

        Tct = np.sum(train_matrix_data_c, axis=1) + 1  # Add-one smoothing
        cond_probs[:, c] = Tct / np.sum(Tct)

    return priors, cond_probs


def apply_multinomial_nb(test_matrix_data, priors, cond_probs):
    scores = np.zeros((len(test_matrix_data), len(priors)))
    for i, doc in enumerate(test_matrix_data):
        for c in range(len(priors)):
            score = np.log(priors[c])
            for j, term_freq in enumerate(doc):
                score += term_freq * np.log(cond_probs[j, c])
            scores[i, c] = score
    return np.argmax(scores, axis=1)


# Evaluate classifier
def evaluate_classifier(test_classes, predicted_labels):
    acc = accuracy_score(test_classes, predicted_labels)
    precision = precision_score(test_classes, predicted_labels)
    recall = recall_score(test_classes, predicted_labels)
    f1 = f1_score(test_classes, predicted_labels)
    cm = confusion_matrix(test_classes, predicted_labels)
    return acc, precision, recall, f1, cm


# function to save confusion matrix
def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{filename}.png')
    plt.close()


# function to save evaluation metrics plot
def plot_evaluation_metrics(accuracy, precision, recall, f1, classifier_name, filename):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    plt.plot(metrics, values, marker='o', color='b', linestyle='-')
    plt.title(f'{classifier_name} Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.savefig(f'{filename}.png')
    plt.close()


# function to save evaluation metrics
def save_evaluation_metrics(acc, precision, recall, f1, nb_confusion_matrix, filename):
    with open(f'{filename}.txt', 'w') as file:
        file.write(f'Accuracy: {acc}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n')
        file.write(f'Confusion Matrix: {nb_confusion_matrix}\n')


# main function
def main():
    train_matrix_data, test_matrix_data, train_classes, test_classes, vocabulary = load_data()

    # train_matrix_data_processed, test_matrix_data_processed = preprocess_data(train_matrix_data, test_matrix_data)

    priors, cond_probs = train_multinomial_nb(train_matrix_data, train_classes)

    # Multinomial Naive Bayes classifier
    predicted_labels_nb = apply_multinomial_nb(test_matrix_data.T, priors, cond_probs)

    nb_accuracy, nb_precision, nb_recall, nb_f1, nb_confusion_matrix = evaluate_classifier(test_classes,
                                                                                           predicted_labels_nb)
    #print("Naive Bayes Accuracy:", nb_accuracy)
    #print("Naive Bayes Confusion Matrix:")
    #print(nb_confusion_matrix)

    # Save Naive Bayes results
    save_confusion_matrix(nb_confusion_matrix, 'Confusion Matrix - Naive Bayes', 'naive_bayes_confusion_matrix')
    save_evaluation_metrics(nb_accuracy, nb_precision, nb_recall, nb_f1, nb_confusion_matrix, 'NB')
    plot_evaluation_metrics(nb_accuracy, nb_precision, nb_recall, nb_f1, 'Naive Bayes', 'naive_bayes_metrics')

    # k-Nearest Neighbors classifier
    k = 5
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(train_matrix_data.T, train_classes)

    predicted_labels_knn = knn_classifier.predict(test_matrix_data.T)
    knn_accuracy, knn_precision, knn_recall, knn_f1, knn_confusion_matrix = evaluate_classifier(test_classes,
                                                                                                predicted_labels_knn)
    #print("k-Nearest Neighbors Accuracy:", knn_accuracy)
    #print("k-Nearest Neighbors Confusion Matrix:")
    #print(knn_confusion_matrix)

    # Save k-Nearest Neighbors results
    save_confusion_matrix(knn_confusion_matrix, 'Confusion Matrix - k-Nearest Neighbors',
                          'k_nearest_neighbors_confusion_matrix')
    save_evaluation_metrics(knn_accuracy, knn_precision, knn_recall, knn_f1, knn_confusion_matrix, 'KNN')
    plot_evaluation_metrics(knn_accuracy, knn_precision, knn_recall, knn_f1, 'k-Nearest Neighbors', 'knn_metrics')

    # Support Vector Machine classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(train_matrix_data.T, train_classes)

    predicted_labels_svm = svm_classifier.predict(test_matrix_data.T)
    svm_accuracy, svm_precision, svm_recall, svm_f1, svm_confusion_matrix = evaluate_classifier(test_classes,
                                                                                                predicted_labels_svm)
    #print("Support Vector Machine Accuracy:", svm_accuracy)
    #print("Support Vector Machine Confusion Matrix:")
    #print(svm_confusion_matrix)

    # Save Support Vector Machine results
    save_confusion_matrix(svm_confusion_matrix, 'Confusion Matrix - Support Vector Machine',
                          'support_vector_machine_confusion_matrix')
    save_evaluation_metrics(svm_accuracy, svm_precision, svm_recall, svm_f1, svm_confusion_matrix, 'SVM')
    plot_evaluation_metrics(svm_accuracy, svm_precision, svm_recall, svm_f1, 'Support Vector Machine', 'svm_metrics')


if __name__ == "__main__":
    main()
