# 0. IMPORT REQUIRE PACKAGES
import pandas as pd
import nltk
import re
import numpy as np
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

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
VOCAB_FILE = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=False),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

encodings = encoder.encode(token_sents_summary)
print(encodings)