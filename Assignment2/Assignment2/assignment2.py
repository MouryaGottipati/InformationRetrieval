import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Loading term-document matrices
trainingDataMatrix = np.loadtxt('./2NewsGroups/trainMatrixModified.txt').transpose()
testingDataMatrix = np.loadtxt('./2NewsGroups/testMatrixModified.txt').transpose()

# Loading class labels
trainingDataClasses = np.loadtxt('./2NewsGroups/trainClasses.txt')[:,1]
testingDataClasses = np.loadtxt('./2NewsGroups/testClasses.txt')[:,1]


def writeMetrics(pred,test,type) :
    with open(f'NBMB_{type}.txt','w') as wrt :
        wrt.write(f'Accuracy : {accuracy_score(test,pred)}\n')
        wrt.write(f'Precision : {precision_score(test,pred)}\n')
        wrt.write(f'Recall : {recall_score(test,pred)}\n')
        wrt.write(f'F1 : {f1_score(test,pred)}\n')
        wrt.write(f'Confusion Matrix : \n{confusion_matrix(test,pred)}\n')
    metrics = [accuracy_score(test,pred,),precision_score(test,pred,),recall_score(test,pred,),f1_score(test,pred,)]
    headers = ['Accuracy','Precision','Recall','F1_score']
    plt.figure()
    plt.plot(headers,metrics)

    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Result')
    plt.title('Evaluation Metrics for ' + type)
    plt.savefig(f'NBMB_{type}_lineplot.png')
    plt.figure(figsize=(8,6))

    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(test,pred,)).plot()
    display.plot()
    plt.title('Confusion Matrix - ' + type)
    plt.savefig(f'NBMB_{type}_confusion_matrix.png')


def trainBernoulliNB(trainingDataMatrix,trainingDataLabels) :
    # Extracting the vocabulary from training set (D).
    vocabulary = np.arange(trainingDataMatrix.shape[1])

    # Counting the total number of documents (N).
    numberOfDocs = trainingDataMatrix.shape[0]

    # Initializing priors and conditional probabilities
    classes = [0,1]
    priors = np.zeros(len(classes))
    conditionalProbabilities = np.zeros((len(vocabulary),len(classes)))

    for c in classes :

        # Counting the number of documents in class c (Nc)
        class_indices = np.where(trainingDataLabels == c)[0]
        Nc = len(class_indices)

        # Calculating the prior probability of class c.
        priors[c] = Nc / numberOfDocs

        for t in range(len(vocabulary)) :
            # Counting the number of documents in class c where term t appears (Dct).
            Dct = np.sum(trainingDataMatrix[class_indices,t] != 0)

            # Calculating the conditional probability for each term t in class c with add-one smoothing.
            conditionalProbabilities[t,c] = (Dct + 1) / (Nc + 2)

    return vocabulary,priors,conditionalProbabilities


def applyBernoulliNB(doc,vocab,priors,conditionalProb) :
    # Extracting the tokens
    tokens = np.zeros(len(vocab))
    for term in range(len(vocab)) :
        if term < len(doc) and doc[term] > 0 :
            tokens[term] = 1

    classes = [0,1]
    # Initializing a score for each class
    scores = np.zeros(len(classes))

    for c in classes :
        # Starting with the log of the prior probability of c
        scores[c] = np.log(priors[c])

        for t in range(len(vocab)) :
            if tokens[t] == 1 :
                scores[c] += np.log(conditionalProb[t,c])
            else :
                scores[c] += np.log(1 - conditionalProb[t,c])

    # Determining the class with the highest score
    classificationResult = np.argmax(scores)
    return classificationResult


# Training the classifier
vocabulary,priors,conditionalProbabilities = trainBernoulliNB(trainingDataMatrix,trainingDataClasses)

# Applying the classifier to all test documents
manualPredictions = []
for i in range(testingDataMatrix.shape[0]) :
    manualPredictions.append(applyBernoulliNB(testingDataMatrix.T[:,i],vocabulary,priors,conditionalProbabilities))

manualPredictions = np.array(manualPredictions)

writeMetrics(manualPredictions,testingDataClasses,'manual')

# Initializing and training the Bernoulli Naive Bayes classifier
bnb = BernoulliNB()
bnb.fit(trainingDataMatrix,trainingDataClasses)

# Generating predictions on the test set
scikitPredictions = bnb.predict(testingDataMatrix)

writeMetrics(scikitPredictions,testingDataClasses,'scikit')
