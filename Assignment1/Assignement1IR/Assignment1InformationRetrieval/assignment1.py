import pprint
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import numpy as np
import matplotlib.pyplot as plt


#Extracting Text and ID's from the Documents
def idsTextExtraction(path) :
    ids = []
    text = []
    currentId = None
    currentText = []
    with open(path,"r") as file :
        for line in file.readlines() :
            line = line.strip()
            if line.startswith(".I") :
                if currentId is not None :
                    ids.append(currentId)
                    text.append(' '.join(currentText))
                currentId = int(line[3 :])
                currentText = []
            elif line.startswith(".W") :
                currentText = []
                continue
            elif currentId is not None :
                currentText.append(line)
        if currentId is not None :
            ids.append(currentId)
            text.append(' '.join(currentText))

        cleanedText = []
        for content in text :
            cleanedContent = ''.join(char if char.isalnum() else ' ' for char in content)
            cleanedText.append(' '.join(cleanedContent.split()))
        for i,docs in enumerate(cleanedText) :
            cleanedText[i] = ' '.join(word.lower() for word in docs.split() if word.lower() not in ENGLISH_STOP_WORDS)
    return ids,cleanedText


# Extracting qrels.txt file to extract query id and docID relevance judgments
def qrelsExtraction(path) :
    qrelsMappings = {}
    for x in open('qrels.text','r') :
        queryId,query,_,_ = map(int,x.strip().split())
        if queryId - 1 not in qrelsMappings :
            qrelsMappings[queryId - 1] = []
        qrelsMappings[queryId - 1].append(query - 1)
    return qrelsMappings


# Part-2: New Custom TF-IDF Vectorizer
class CustomTfidfVectorizer(CountVectorizer) :
    def __init__(self) :
        super().__init__(stop_words='english',lowercase=True)

    def fit_transform(self,contents,y=None) :
        docTermMatrix = super().fit_transform(contents)
        docTermMatrix = docTermMatrix.astype(float)
        tf = docTermMatrix.copy()
        doc_lengths = np.array(docTermMatrix.sum(axis=1)).flatten()

        for doc_idx in range(docTermMatrix.shape[0]) :
            doc_length = doc_lengths[doc_idx]
            if doc_length == 0 :
                doc_length = 1
            for term_idx in docTermMatrix[doc_idx].indices :
                term_count = tf[doc_idx,term_idx]
                if term_count == 0 :
                    term_count = 1
                tf[doc_idx,term_idx] = (1 + np.log(term_count)) / (1 + np.log(doc_length))

        df = np.diff(docTermMatrix.tocsc().indptr)
        idf = np.reciprocal(df.astype(float))
        idf[df == 0] = 1
        tf_idf = tf.multiply(idf)
        return tf_idf


def similarityCalculation(docWeightsVector,queryWeightsVector,metric='cosine') :
    if metric == 'cosine' :
        return cosine_similarity(queryWeightsVector,docWeightsVector)
    elif metric == 'euclidean' :
        return euclidean_distances(queryWeightsVector,docWeightsVector)


def retrievingTop10RelevantDocs(similarityMatrix,metric='cosine') :
    if metric == 'cosine' :
        relevantDocs = np.argsort(similarityMatrix,axis=1)[:,: :-1][:,:10]
    elif metric == 'euclidean' :
        relevantDocs = np.argsort(similarityMatrix,axis=1)[:,:10]
    return relevantDocs


def metricsCalculation(top10RelevantDocs,qrels) :
    precisions,recalls,fScores = [],[],[]

    i = 0
    for queryID in range(0,225) :
        relevant_docs = set(qrels[queryID])
        retrieved_docs = set(top10RelevantDocs[i])

        i += 1
        tp = len(relevant_docs & retrieved_docs)
        fp = len(retrieved_docs - relevant_docs)
        fn = len(relevant_docs - retrieved_docs)

        if (tp + fp) > 0 :
            precision = float(tp / (tp + fp))
        else :
            precision = 0
        if (tp + fn) > 0 :
            recall = float(tp / (tp + fn))
        else :
            recall = 0

        if (precision + recall) > 0 :
            f_score = float(2 * float(precision * recall) / float(precision + recall))
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
            'f' : {'cos' : (float(np.mean(scores[0][2])),float(np.max(scores[0][2]))),
                   'euc' : (float(np.mean(scores[1][2])),float(np.max(scores[1][2])))},
            'p' : {'cos' : (float(np.mean(scores[0][0])),float(np.max(scores[0][0]))),
                   'euc' : (float(np.mean(scores[1][0])),float(np.max(scores[1][0])))},
            'r' : {'cos' : (float(np.mean(scores[0][1])),float(np.max(scores[0][1]))),
                   'euc' : (float(np.mean(scores[1][1])),float(np.max(scores[1][1])))}
        },
        'TFIDF' : {
            'f' : {'cos' : (float(np.mean(scores[2][2])),float(np.max(scores[2][2]))),
                   'euc' : (float(np.mean(scores[3][2])),float(np.max(scores[3][2])))},
            'p' : {'cos' : (float(np.mean(scores[2][0])),float(np.max(scores[2][0]))),
                   'euc' : (float(np.mean(scores[3][0])),float(np.max(scores[3][0])))},
            'r' : {'cos' : (float(np.mean(scores[2][1])),float(np.max(scores[2][1]))),
                   'euc' : (float(np.mean(scores[3][1])),float(np.max(scores[3][1])))}
        }
    }
    return scoresDictionary


def stepsStart() :
    # Documents Extraction
    docIDs,documents = idsTextExtraction("./cran.all")

    # Queries Extraction
    queryIDs,queries = idsTextExtraction("./query.text")

    # Query Document Relevance Mapping
    queryDocRelevanceMapping = qrelsExtraction("./qrels.text")

    # Part-1: Default Binary Vectorizer from Scikit-learn
    binaryVectorizer = CountVectorizer(binary=True,stop_words='english',lowercase=True)
    binaryDoc = binaryVectorizer.fit_transform(documents)
    binaryQ = binaryVectorizer.transform(queries)

    # Part-2: Custom TF-IDF Vectorizer
    CustomTfIdfVectorizer = CustomTfidfVectorizer()
    customTfIdfDoc = CustomTfIdfVectorizer.fit_transform(documents)
    customTfIdfQ = CustomTfIdfVectorizer.transform(queries)

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
    cosineBinaryMetrics = metricsCalculation(cosineBinaryTop10,queryDocRelevanceMapping)
    euclideanBinaryMetrics = metricsCalculation(euclideanBinaryTop10,queryDocRelevanceMapping)
    cosineCustomTfIdfMetrics = metricsCalculation(customCosineTfIdfTop10,queryDocRelevanceMapping)
    euclideanCustomTfIdfMetrics = metricsCalculation(customEuclideanTfIdfTop10,queryDocRelevanceMapping)

    # # Saving bar plots for all metrics and models along with various distance type
    for vectorizerType in ['Binary','CustomTfIdf'] :
        for metrics in [cosineBinaryMetrics,euclideanBinaryMetrics,cosineCustomTfIdfMetrics,
                        euclideanCustomTfIdfMetrics] :
            if metrics == cosineBinaryMetrics or metrics == cosineCustomTfIdfMetrics :
                distanceType = "Cosine"
            else :
                distanceType = "Euclidean"
            precisions,recalls,fScores = metrics
            savingBarPlots(precisions,'Precision',vectorizerType,distanceType)
            savingBarPlots(recalls,'Recall',vectorizerType,distanceType)
            savingBarPlots(fScores,'F-score',vectorizerType,distanceType)

    # Displaying metrics
    displayScores = dictionary(
        [cosineBinaryMetrics,euclideanBinaryMetrics,cosineCustomTfIdfMetrics,euclideanCustomTfIdfMetrics])
    pprint.pprint(displayScores)


stepsStart()
