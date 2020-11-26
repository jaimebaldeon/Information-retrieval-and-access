# importing required modules 
from zipfile import ZipFile 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# specifying the zip file name 
file_name = "Ficheros(html).zip"

# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
	# extracting the names of the files in the zip
	file_names = zip.namelist()
	# extracting all the files 
	print('Extracting all the files now...') 
	zip.extractall() 
	print('Done!') 


clean_data = []
print('Cleaning raw data...')
for file_name in file_names:
  with open(file_name, 'r') as file:
      rawdata = file.read().replace('\n', '')
      clean_script = re.compile('<script.*?</script>')
      clean_script_data = re.sub(clean_script, '', rawdata)
      clean_htmltags = re.compile('<.*?>')
      clean_htmltags_data = re.sub(clean_htmltags, ' ', clean_script_data)
      clean_data.append(re.sub('\s+',' ',clean_htmltags_data))
print('Done!')


corpus = clean_data
corpus

vectorizer = CountVectorizer()
# tokenization
matriz_tf = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()

# Matrix with token occurrences 
matriz_tf.toarray()

# Analysis of corpus documents
analyze = vectorizer.build_analyzer()
for documento in corpus: print(analyze(documento))

# Create bigram vectorizer (more semantic information)
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), 
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()

bmatriz_tf = bigram_vectorizer.fit_transform(corpus)
bigram_vectorizer.get_feature_names()

query = [
    "What video game won Spike's best driving game award in 2006?"
]
query

query_tf = vectorizer.transform(query)
bquery_tf = bigram_vectorizer.transform(query)
query_tf.toarray()

# Similarity with Scalar Product TF

num_files = matriz_tf.get_shape()[0]
q = query_tf.toarray().flatten()
scalar_prod_TF = []
for i in range(num_files):
  doc = matriz_tf.getrow(i).toarray().flatten()
  scalar_prod_TF.append(q @ doc )
scalar_prod_TF

# Similarity with Scalar Product TF

num_files = bmatriz_tf.get_shape()[0]
q = bquery_tf.toarray().flatten()
scalar_prod_TF = []
for i in range(num_files):
  doc = bmatriz_tf.getrow(i).toarray().flatten()
  scalar_prod_TF.append(q @ doc )
scalar_prod_TF

# Similarity with Cosine TF

print(cosine_similarity(query_tf, matriz_tf))
cosine_similarity(bquery_tf, bmatriz_tf)

tfidf_vectorizer = TfidfVectorizer()
matriz_tfidf = tfidf_vectorizer.fit_transform(corpus)
# Token weights in each document
matriz_tfidf.toarray()

tfidf_bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), 
                                    token_pattern=r'\b\w+\b', min_df=1)
bmatriz_tfidf = tfidf_bigram_vectorizer.fit_transform(corpus)

# Total tokens' weight
tfidf_vectorizer.idf_

tfidf_vectorizer.get_feature_names()

query_tfidf = tfidf_vectorizer.transform(query)
bquery_tfidf = tfidf_bigram_vectorizer.transform(query)
query_tfidf.toarray()

# Similarity with Scalar Product TF IDF
num_files = matriz_tfidf.get_shape()[0]

# Tf*idf computation of words of the query 
q_tfidf = query_tf.toarray().flatten() * tfidf_vectorizer.idf_.flatten()  

scalar_prod_TFIDF = []
for i in range(num_files):
  doc_tfidf = matriz_tf.getrow(i).toarray().flatten() * tfidf_vectorizer.idf_.flatten()
  scalar_prod_TFIDF.append(q_tfidf @ doc_tfidf )
scalar_prod_TFIDF

# Similarity with Scalar Product TF IDF
num_files = bmatriz_tfidf.get_shape()[0]

# Tf*idf computation of words of the query 
q_tfidf = bquery_tf.toarray().flatten() * tfidf_bigram_vectorizer.idf_.flatten()  

scalar_prod_TFIDF = []
for i in range(num_files):
  doc_tfidf = bmatriz_tf.getrow(i).toarray().flatten() * tfidf_bigram_vectorizer.idf_.flatten()
  scalar_prod_TFIDF.append(q_tfidf @ doc_tfidf )
scalar_prod_TFIDF

# Similarity with Cosine TF IDF

print(cosine_similarity(query_tfidf, matriz_tfidf))
cosine_similarity(bquery_tfidf, bmatriz_tfidf)
