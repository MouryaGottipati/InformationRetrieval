Team
Mourya Gottipati    -    U01099603    -    gottipati.22@wright.edu
Niranjan Kumar Ramineni    -    U01065247    -    ramineni.15@wright.edu

Python Version - 3.10.11

Application
This is a Search Engine Application using Information Retrieval concept that are covered in the course.
In this we process the Cranfield Dataset which consists of Cran.all consists of 1400 documents,query.txt consists of 225 queries,
qrels.txt consist of Documents relevant 225 queries.
Mainly concentrating on Two models they are default Binary Vectorizer and Custom Vectorizer which inherited properties of CountVectorizer().
In this Cosine Similarity and Euclidean Distances are used for Retrieving Top 10 Relevant document for each of the query.
Based on the retrieved documents and the relevant documents from qrels.txt file Precision,Recall,F-score are calculated for each model
with each Cosine Similarity/distance type.
Resulting Graphs from these Values are made into plots and stored in the working directory.
Output Results being displayed in the console which consists of mean and maximum values for each metric.

Launching Application
1.) open terminal
2.) Navigate to working directory
3.) run command  -  "python assignment1.py"

After running this file the Output Results will be displayed in the console and
resultant graphs will be stored in the working directory

Libraries Used

1.) pprint: Pretty-print for Python data structures.
2.) sklearn.feature_extraction.text: CountVectorizer and ENGLISH_STOP_WORDS for text vectorization.
3.) sklearn.metrics.pairwise: cosine_similarity and euclidean_distances for calculating similarity/distance metrics.
4.) numpy: For numerical operations.
5.) matplotlib.pyplot: For plotting.



