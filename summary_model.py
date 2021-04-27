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

# 2. CLEANING TEXT

  # Get stopwords
stopwords = nltk.corpus.stopwords.words('english')

def lite_split_sentence(content):
  contents_parsed = re.sub('\n', ' ', content) #Đổi các ký tự xuống dòng thành chấm câu
  contents_parsed = re.sub(r'\([^)]*\)', '', contents_parsed)
  contents_parsed = re.sub(r'\s+', ' ', contents_parsed)
  

  sentences = nltk.sent_tokenize(contents_parsed)

  return sentences


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

# 3. CONVERT TEXT => SENTENCES
def text_to_sents(text):
  sents = nltk.sent_tokenize(text)
  return [re.sub(r'\.','',sent) for sent in sents]

w2v = Word2Vec.load("en_txt\en.bin")

vocab = w2v.wv.index_to_key

def Summary(text, brif=0.3):

  # 1. clean text
  new_text = text_strip(text)

  # 2. to sentences
  sentences = lite_split_sentence(text)
  sents_of_text = text_to_sents(new_text)

  # 3. tokenized sentences
  token_sents_text = [nltk.word_tokenize(sent) for sent in sents_of_text]

  # 4. get vectors
  X = []

  for sent in token_sents_text:
    sent_vec = np.zeros((300))
    for word in sent:
      if word in vocab:
        sent_vec+=w2v.wv[word]
    X.append(sent_vec)

  # 5. cluster text
  n_clusters = int(len(X)*brif)

  kmeans = KMeans(n_clusters=n_clusters)
  kmeans = kmeans.fit(X)

  # 6. build a summary text
  avg = []
  for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append((np.mean(idx)))
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
  ordering = sorted(range(n_clusters), key=lambda k: avg[k])
  summary = '\n'.join([sentences[closest[idx]] for idx in ordering])

  return summary


# summary_text = Summary(text=text, brif=0.3)
# print(summary_text)

