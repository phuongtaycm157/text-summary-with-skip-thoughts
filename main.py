# 0. IMPORT REQUIRE PACKAGES
import pandas as pd
import nltk
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# 1. READING ARTICLE FROM DATASET
dataset = pd.read_csv('crawl_textsummary.csv')

text = dataset.iloc[0].content
summary = dataset.iloc[0].summarization

# 2. CLEANING TEXT

  # Get stopwords
stopwords = nltk.corpus.stopwords.words('english')

def text_strip(text):
  # Loại bỏ các ký tự đặt biệt, khoản trắng thừa, giữ .
  newString = text.lower()
  newString = re.sub(r'\([^)]*\)', '', newString)
  newString = re.sub('"','', newString)
  newString = re.sub("[^a-zA-Z\.]", " ", newString) 
  newString = re.sub(r'\s+', ' ', newString)
  
  # Loại bỏ stopwords
  no_stopwords = []
  for word in nltk.word_tokenize(newString):
    if word not in stopwords:
      no_stopwords.append(word)

  new_no_stopwords_string = ' '.join(no_stopwords)
  new_no_stopwords_string = re.sub(r'\s\.', '.',new_no_stopwords_string)

  return new_no_stopwords_string

text = text_strip(text)
summary = text_strip(summary)

# 3. CONVERT TEXT => SENTENCES
def text_to_sents(text):
  sents = nltk.sent_tokenize(text)
  return [re.sub(r'\.','',sent) for sent in sents]

sents_of_text = text_to_sents(text)
sents_of_summary = text_to_sents(summary)

# 4. TOKENIZED SENTENCES
token_sents_text = [nltk.word_tokenize(sent) for sent in sents_of_text]
token_sents_summary = [nltk.word_tokenize(sent) for sent in sents_of_summary]

# 5. SENTENCE VECTOR
w2v = Word2Vec.load("en_txt\en.bin")

vocab = w2v.wv.index_to_key

X = []

for sent in token_sents_text:
  sent_vec = np.zeros((300))
  for word in sent:
    if word in vocab:
      sent_vec+=w2v.wv[word]
  X.append(sent_vec)

# y = []

# for sent in token_sents_summary:
#   sent_vec = np.zeros((300))
#   for word in sent:
#     if word in vocab:
#       sent_vec+=wv.wv[word]
#   y.append(sent_vec)

n_clusters = len(X)//3

kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)

avg = []
for j in range(n_clusters):
  idx = np.where(kmeans.labels_ == j)[0]
  avg.append((np.mean(idx)))

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary_result = '. '.join([sents_of_text[closest[idx]] for idx in ordering])
print(summary)
print('=================================')
print(summary_result)