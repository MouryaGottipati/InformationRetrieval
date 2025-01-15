"""
Implementation of Various classifiers using Scikit-learn and Implementing NaiveBayes from scratch
Below is our implementation and the data setup.
Input files - trainMatrixModified.txt, trainClasses.txt, testMatrixModified.txt, testClasses.txt
Output files - KNN.txt, SVM.txt, NB.txt, 
               KNN-Graphs.jpg, SVM-Graphs.png, NB-Graphs.png,
               KNN-CM.png,SVM-CM.png, NB-CM.png
"""
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, \
    recall_score
import math


def write_metrics(pred, test, model):
    with open(model+'.txt', 'w') as wrt:
        wrt.write(f'Accuracy : {accuracy_score(pred, test)}\n')
        wrt.write(f'Recall : {recall_score(pred, test)}\n')
        wrt.write(f'Precision : {precision_score(pred, test)}\n')
        wrt.write(f'Confusion Matrix : \n{confusion_matrix(pred, test)}\n')
    metrics = [accuracy_score(pred, test), recall_score(pred, test), precision_score(pred, test), f1_score(pred, test)]
    headers = ['Accuracy', 'Recall', 'Precision', 'F1_score']
    plt.figure()
    plt.bar(headers, metrics)

    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Result')
    plt.title('Evaluation Metrics for '+model)
    plt.savefig(f'{model}_EM.png')
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(pred, test)).plot()
    plt.savefig(model + '_CM.png')


print('Reading files using numpy')


trainData_Matrix = np.loadtxt('trainMatrixModified.txt').transpose()
trainData_Classes = np.loadtxt('trainClasses.txt')
testData_Matrix = np.loadtxt('testMatrixModified.txt').transpose()
testData_Classes = np.loadtxt('testClasses.txt')

print('Started Classification for SVM')

classifier_svm = SVC(kernel='linear').fit(trainData_Matrix, trainData_Classes[:, 1])
p_svm = classifier_svm.predict(testData_Matrix)

write_metrics(p_svm,testData_Classes[:, 1], 'SVM')

print('Started Classification for KNN')

classifier_knn = KNeighborsClassifier(n_neighbors=4).fit(trainData_Matrix, trainData_Classes[:, 1])
p_knn = classifier_knn.predict(testData_Matrix)

write_metrics(p_knn,testData_Classes[:, 1], 'KNN')


def TRAINMULTINOMIALNB(dataset_train,labels_train):
    samples_WIN=np.count_nonzero(labels_train==0)
    samples_Hock=np.count_nonzero(labels_train)
    WinConcatinates=[]
    HockConcatinates=[]
    i=0
    while i <labels_train.size:
        if(labels_train[i]==0.00):
            WinConcatinates.append(dataset_train[i,:])
        else:
            HockConcatinates.append(dataset_train[i,:])
        i+=1
    ColumnSumInWIn=np.sum(WinConcatinates,0)
    ColumnSumInHock=np.sum(HockConcatinates,0)
    return [((ColumnSumInWIn+1)/np.sum((ColumnSumInWIn+1))),((ColumnSumInHock+1)/np.sum((ColumnSumInHock+1)))],[(samples_WIN/dataset_train.shape[0]),(samples_Hock/dataset_train.shape[0])]
def APPLYMULTINOMIALNB(testDataSet,Conditional_Probability_Train,priorTrain):
    PriorTrainScoreWin=(math.log(priorTrain[0]))
    PriorTrainScoreHock=(math.log(priorTrain[1]))
    testPreds=[]
    for testDataSetRecord in testDataSet:
        TempVarWin=PriorTrainScoreHock
        TempVarHock=PriorTrainScoreWin
        for i in range(len(testDataSetRecord)):
            if(testDataSetRecord[i]>=1.0):
                TempVarHock+=math.log(Conditional_Probability_Train[0][i])
                TempVarWin+=math.log(Conditional_Probability_Train[1][i])
        testPreds.append(1 if TempVarWin>TempVarHock else 0)
    return testPreds


cond_prob, p_prob = TRAINMULTINOMIALNB(trainData_Matrix, trainData_Classes[:, 1])
p_nb = APPLYMULTINOMIALNB(testData_Matrix,cond_prob,p_prob)
write_metrics(p_nb,testData_Classes[:, 1], 'NB')




