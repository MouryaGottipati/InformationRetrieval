import pprint
import re
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import numpy as np
import matplotlib.pyplot as plt


# Extracting Document content
def documentsIDsExtraction(path) :
    file = open(path,"r")
    text = file.read()

    # Stripping out docIDs and storing into a list
    docIDs = re.findall(r'\.I\s+(\d+)',text)
    docIDs = [int(id) for id in docIDs]

    # Stripping out each document
    strippingEachDocumentContents = re.findall(r'\.I\s+\d+\n([\s\S]+?)(?=\n\.I\s+\d+|\Z)',text)

    # Stripping out .W content in each document
    documentContents = []
    for i in strippingEachDocumentContents :
        match = re.findall(r'\.W([\s\S]+)',i)
        if match :
            documentContents.append(match[0].replace("\n",' ').strip())
        else :
            documentContents.append('')
    return docIDs,documentContents


# Extracting Queries and query IDs
def queriesIDsExtraction(path) :
    file = open(path,"r")
    text = file.read()

    # Stripping out queryIDs and storing into a list """[i for i in range(1,226)]"""
    queryIDs = re.findall(r'\.I\s+(\d+)',text)
    queryIDs = [int(id) for id in queryIDs]

    # Stripping out each query
    strippingEachQueryContents = re.findall(r'\.I\s+\d+\n([\s\S]+?)(?=\n\.I\s+\d+|\Z)',text)

    # Stripping out .W content in each query content
    queryContents = []
    for i in strippingEachQueryContents :
        match = re.findall(r'\.W([\s\S]+)',i)
        if match :
            queryContents.append(match[0].replace("\n",' ').strip())
        else :
            queryContents.append('')
    return queryIDs,queryContents


# Extracting qrels.txt file to extract query id and docID relevance judgments
def qrelsExtraction(path) :
    qrelsMappings = {}
    with open(path,"r") as file :
        for line in file.readlines() :
            qrels = line.strip().split()
            queryID = int(qrels[0])
            relevantDocument = int(qrels[1])

            if queryID in qrelsMappings :
                qrelsMappings[queryID].append(relevantDocument)
            else :
                qrelsMappings[queryID] = [relevantDocument]
    return qrelsMappings


# Part-2: New Custom TF-IDF Vectorizer
class CustomTfidfVectorizer(CountVectorizer) :

    def __init__(self) :
        super().__init__(stop_words='english',lowercase=True)

    def fit_transform(self,raw_documents) :
        X = super().fit_transform(raw_documents)

        # Calculating Custom idf values
        self.idf_ = self.customIdf(X)

        # Calculating weights from above idf values while multiplying with custom term frequency values
        X = self.tfIdfTransform(X)
        return X

    def transform(self,rawDocuments) :
        X = super().transform(rawDocuments)
        X = self.tfIdfTransform(X)
        return X

    def customIdf(self,X) :
        N = X.shape[0]
        df = np.sum(X > 0,axis=0)
        df = np.asarray(df).flatten()
        idf = 1 / df
        idf[np.isinf(idf)] = 1
        return idf

    def tfIdfTransform(self,X) :
        tf = X.copy()

        for i in range(X.shape[0]) :
            row = X.getrow(i)
            rowData = row.data
            rowIndices = row.indices
            docLength = row.sum()
            for idx in range(len(rowData)) :
                j = rowIndices[idx]
                termCount = rowData[idx]
                tf[i,j] = (1 + np.log(termCount if termCount > 0 else 1)) / (
                        1 + np.log(docLength if docLength > 0 else 1))

        for idx in range(tf.data.size) :
            tf.data[idx] *= self.idf_[tf.indices[idx]]

        return tf



def similarityCalculation(docWeightsVector,queryWeightsVector,metric='cosine') :
    if metric == 'cosine' :
        return cosine_similarity(queryWeightsVector,docWeightsVector)
    elif metric == 'euclidean' :
        return euclidean_distances(queryWeightsVector,docWeightsVector)


def retrievingTop10RelevantDocs(similarityMatrix,metric='cosine') :
    if metric == 'cosine' :
        relevantDocs = np.argsort(-similarityMatrix,axis=1)[:,:10]
    elif metric == 'euclidean' :
        relevantDocs = np.argsort(similarityMatrix,axis=1)[:,:10]
    return relevantDocs


def metricsCalculation(top10RelevantDocs,qrels,queryIDs) :
    precisions,recalls,fScores = [],[],[]

    i=0
    for queryID in queryIDs :
        if queryID in qrels.keys():
            relevant_docs = set(qrels[queryID])
        else:
            relevant_docs=set()

        retrieved_docs = set(top10RelevantDocs[i])
        
        i+=1
        tp = len(relevant_docs & retrieved_docs)
        fp = len(retrieved_docs - relevant_docs)
        fn = len(relevant_docs - retrieved_docs)

        if (tp + fp) > 0 :
            precision = tp / (tp + fp)
        else :
            precision = 0
        if (tp + fn) > 0 :
            recall = tp / (tp + fn)
        else :
            recall = 0

        if (precision + recall) > 0 :
            f_score = 2 * (precision * recall) / (precision + recall)
        else :
            f_score = 0

        precisions.append(precision)
        recalls.append(recall)
        fScores.append(f_score)

    return precisions,recalls,fScores


def savingBarPlots(metricValues,metric,vectorizerType,distanceType) :
    plt.figure()
    plt.bar(range(len(metricValues)),metricValues)
    plt.xlabel('Query Index')
    plt.ylabel(metric)
    plt.title(f'{metric} of each query (10 most relevant documents)- {vectorizerType} (using {distanceType})')
    plt.savefig(f'{metric}_{vectorizerType}_{distanceType}.png')
    plt.close()


def dictionary(scores) :
    scoresDictionary = {
        'Binary' : {
            'f' : {'cos' : (np.mean(scores[0][2]),np.max(scores[0][2])),
                   'euc' : (np.mean(scores[1][2]),np.max(scores[1][2]))},
            'p' : {'cos' : (np.mean(scores[0][0]),np.max(scores[0][0])),
                   'euc' : (np.mean(scores[1][0]),np.max(scores[1][0]))},
            'r' : {'cos' : (np.mean(scores[0][1]),np.max(scores[0][1])),
                   'euc' : (np.mean(scores[1][1]),np.max(scores[1][1]))}
        },
        'TFIDF' : {
            'f' : {'cos' : (np.mean(scores[2][2]),np.max(scores[2][2])),
                   'euc' : (np.mean(scores[3][2]),np.max(scores[3][2]))},
            'p' : {'cos' : (np.mean(scores[2][0]),np.max(scores[2][0])),
                   'euc' : (np.mean(scores[3][0]),np.max(scores[3][0]))},
            'r' : {'cos' : (np.mean(scores[2][1]),np.max(scores[2][1])),
                   'euc' : (np.mean(scores[3][1]),np.max(scores[3][1]))}
        }
    }
    return scoresDictionary


def stepsStart() :
    # Documents Extraction
    docIDs,documents = documentsIDsExtraction("./cran.all")
    print(f"length of docs{1} and {2}",len(documents),documents[0])

    # Queries Extraction
    queryIDs,queries = queriesIDsExtraction("./query.text")
    print(f"length of queries {1} and {2}",len(queries),queries[0])
    print(f"query id's{1}",queryIDs[:10])

    # Query Document Relevance Mapping
    queryDocRelevanceMapping = qrelsExtraction("./qrels.text")
    print(f"length of queryDocRelevanceMapping {1} and {2}",len(queryDocRelevanceMapping.keys()),queryDocRelevanceMapping[1])
    print(queryDocRelevanceMapping)

    # Part-1: Default Binary Vectorizer from Scikit-learn
    binaryVectorizer = CountVectorizer(binary=True,stop_words=list(ENGLISH_STOP_WORDS),lowercase=True)
    binaryDoc = binaryVectorizer.fit_transform(documents)
    binaryQ = binaryVectorizer.transform(queries)

    # Part-2: Custom TF-IDF Vectorizer
    customTfIdfVectorizer = CustomTfidfVectorizer()
    customTfIdfDoc = customTfIdfVectorizer.fit_transform(documents)
    customTfIdfQ = customTfIdfVectorizer.transform(queries)

    # Calculating similarity for binary vectorizer
    consineSimBinary = similarityCalculation(binaryDoc,binaryQ)
    euclideanDistBinary = similarityCalculation(binaryDoc,binaryQ,metric='euclidean')

    # Calculating similarity for custom TF-IDF vectorizer
    cosineSimCustomTfIdf = similarityCalculation(customTfIdfDoc,customTfIdfQ)
    euclideanDistCustomTfIdf = similarityCalculation(customTfIdfDoc,customTfIdfQ,metric='euclidean')

    # Getting top 10 relevant document ids  for each query
    cosineBinaryTop10 = retrievingTop10RelevantDocs(consineSimBinary)
    euclideanBinaryTop10 = retrievingTop10RelevantDocs(euclideanDistBinary,metric='euclidean')
    customCosineTfIdfTop10 = retrievingTop10RelevantDocs(cosineSimCustomTfIdf)
    customEuclideanTfIdfTop10 = retrievingTop10RelevantDocs(euclideanDistCustomTfIdf,metric='euclidean')

    # Calculating metrics for both models with respect to Above obtained relevant docs and
    # Provided Query and respective relevant doc ids from qrels
    cosineBinaryMetrics = metricsCalculation(cosineBinaryTop10,queryDocRelevanceMapping,queryIDs)
    euclideanBinaryMetrics = metricsCalculation(euclideanBinaryTop10,queryDocRelevanceMapping,queryIDs)
    cosineCustomTfIdfMetrics = metricsCalculation(customCosineTfIdfTop10,queryDocRelevanceMapping)
    euclideanCustomTfIdfMetrics = metricsCalculation(customEuclideanTfIdfTop10,queryDocRelevanceMapping)

    # # Saving bar plots for all metrics and models along with various distance type
    for metric,allMetrics in zip(['Precision','Recall','F-score'],
                                 [cosineBinaryMetrics,euclideanBinaryMetrics,
                                  cosineCustomTfIdfMetrics,euclideanCustomTfIdfMetrics]) :
        vectorizerTypes = ['Binary','Binary','CustomTfIdf','CustomTfIdf']
        distanceTypes = ['Cosine','Euclidean','Cosine','Euclidean']

        for vectorizerType,distanceType,metrics in zip(vectorizerTypes,distanceTypes,
                                                       [allMetrics,allMetrics,allMetrics,allMetrics]) :
            precisions,recalls,fScores = metrics
            savingBarPlots(precisions,'Precision',vectorizerType,distanceType)
            savingBarPlots(recalls,'Recall',vectorizerType,distanceType)
            savingBarPlots(fScores,'F-score',vectorizerType,distanceType)

    # Displaying metrics
    displayScores = dictionary(
        [cosineBinaryMetrics,euclideanBinaryMetrics,cosineCustomTfIdfMetrics,euclideanCustomTfIdfMetrics])
    pprint.pprint(displayScores)
stepsStart()
