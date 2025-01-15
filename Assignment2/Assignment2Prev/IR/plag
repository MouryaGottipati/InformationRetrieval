import os

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# stopwords and punkt download
nltk.download('stopwords')
nltk.download('punkt')

doc_id_name = {}


# loading documents
def load_cran_all(directory):
    doc = {}
    with open(os.path.join(directory), "r") as file:
        lines = [line.strip() for line in file.readlines()]
        document_id = ""
        doc_id = 0
        is_document_id = False
        is_text = False
        for line in lines:
            if line.startswith(".I"):
                doc_id = int(line.replace(".I ", ""))
                is_text = False
            if line.startswith(".T"):
                is_text = False
                is_document_id = True
                document_id = ""
            if line.startswith(".A"):
                is_document_id = False
            if line.startswith(".W"):
                is_text = True
            if is_document_id:
                document_id = document_id + "" + line
                document_id = document_id.replace(".T", "")
            if is_text:
                doc_id_name[document_id] = doc_id
                if document_id in doc.keys():
                    doc[document_id] = doc[document_id].replace(".W", "") + "" + line
                else:
                    doc[document_id] = line
    return doc


# loading queries
def load_browsed_queries(filename):
    with open(filename, "r") as file:
        browsed_queries = []
        lines = [line.strip() for line in file.readlines()]
        is_query = False
        browse_query = ""
        for line in lines:
            if line.startswith(".W"):
                is_query = True
                browse_query = ""
            if line.startswith(".I"):
                is_query = False
                if len(browse_query) != 0:
                    browsed_queries.append(browse_query.replace(".W", ""))
            if is_query:
                browse_query = browse_query + "" + line
    return browsed_queries


# loading qrels
def load_qrels(filename):
    with open(filename, "r") as file:
        qrels = [line.strip().split() for line in file.readlines()]
    return qrels


# Preprocessing doc and query texts
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # getting stopwords
    tokens = word_tokenize(text.lower())  # converting all text to lower case
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]  # removing all stop words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]  # handling synonyms and lemitization
    return tokens


# processing the query before for getting relevant docs for speed up
def process_query(query, inverted_index):
    tokens = preprocess_text(query)
    relevant_documents = set()
    for token in tokens:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])
    return relevant_documents


# Indexing the documents
def build_inverted_index(doc):
    indecies = defaultdict(list)
    for doc_id, text in doc.items():
        tokens = preprocess_text(text)
        for token in tokens:
            indecies[token].append(doc_id)
    return indecies


# Ranking docs with given query only top 10 results for query was given
def rank_docs(query, docs, vectorizer, measure):
    if vectorizer == "TF-IDF":
        use_vectorizer = TfidfVectorizer()
    elif vectorizer == "Binary":
        use_vectorizer = CountVectorizer(binary=True)

    matrix = use_vectorizer.fit_transform(docs)
    query_vector = use_vectorizer.transform([query])
    if measure == "cosine":
        similarities = cosine_similarity(query_vector, matrix)  # for cosine similarities
    elif measure == "manhattan":
        similarities = pairwise_distances(query_vector, matrix, metric='manhattan')  # for manhattan similarities

    ranked_indices = similarities.argsort()[0][:-11:-1]  # sorting and getting top 10 results
    ranked_documents = [(list(docs)[i], similarities[0][i]) for i in ranked_indices]
    return ranked_documents


# loading all required data
documents_directory = "cran.all"
docs = load_cran_all(documents_directory)
queries_filename = "query.text"
browsed_queries = load_browsed_queries(queries_filename)
qrels_filename = "qrels.text"
qrels = load_qrels(qrels_filename)

inverted_index_tokens = build_inverted_index(docs)
vectorizers = ["Binary", "TF-IDF"]
measures = ["cosine", "manhattan"]
result = {}


def get_top_docs(vectorizer, measure):
    top_documents_list = []
    for query in browsed_queries:
        relevant_docs = process_query(query, inverted_index_tokens)
        ranked_documents = rank_docs(query, relevant_docs, vectorizer, measure)
        top_documents_list.append(ranked_documents)
    return top_documents_list


for vectorizer in vectorizers:
    for measure in measures:
        key = f"{vectorizer}_{measure}"
        result[key] = get_top_docs(vectorizer, measure)


def calculate_metrics(retrieved_docs, relevant_docs):
    tp = len(set(retrieved_docs).intersection(relevant_docs))
    p = tp / len(retrieved_docs)
    r = tp / len(relevant_docs)
    if p + r == 0:
        f = 0
    else:
        f = 2 * (p * r) / (p + r)
    return p, r, f


metrics = {}
for key, retrieved_documents in result.items():
    vectorizer, measure = key.split("_")
    precision_list = []
    recall_list = []
    f_score_list = []
    for i, retrieved_docs in enumerate(retrieved_documents):
        relevant_docs = [int(doc[1]) for doc in qrels if int(doc[0]) == i + 1]
        retrieved_docs_ids = [doc_id_name[ids] for ids, score in retrieved_docs]
        precision, recall, f_score = calculate_metrics(retrieved_docs_ids, relevant_docs)
        precision_list.append(precision)
        recall_list.append(recall)
        f_score_list.append(f_score)
    metrics[key] = {"precision": precision_list, "recall": recall_list, "f_score": f_score_list}


def plot_metrics(metrics, vec, measure):
    metrics_names = ["precision", "recall", "f_score"]
    for metric_name in metrics_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(metrics[metric_name])), metrics[metric_name])
        ax.set_xlabel("Query ID")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} - {vec.upper()} - {measure.capitalize()}")
        plt.tight_layout()
        plt.savefig(f"{metric_name[0]}_{vec}_{measure.lower()}.png")


for vect in vectorizers:
    for measure in measures:
        key = f"{vect}_{measure}"
        plot_metrics(metrics[key], vect, measure)

mean_max_metrics = {}
precision_min_max = {}
recall_min_max = {}
f_score_min_max = {}
for key, metric_data in metrics.items():
    vect, measure = key.split("_")
    precision_min_max[measure] = {np.mean(metric_data["precision"]), np.max(metric_data["precision"])}
    recall_min_max[measure] = {np.mean(metric_data["recall"]), np.max(metric_data["recall"]), }
    f_score_min_max[measure] = {np.mean(metric_data["f_score"]), np.max(metric_data["f_score"])}
    mean_max_metrics[vect] = {"f": f_score_min_max, "p": precision_min_max, "r": recall_min_max}

print(str(mean_max_metrics).replace("cosine", "cos").replace("manhattan", "man"))
