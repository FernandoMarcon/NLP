# NPL Basics
import pandas as pd
pd.set_option('display.max_colwidth',100)

#--- Read in semi-structured text data
data = pd.read_csv('data/SMSSpamCollection.tsv', sep = '\t', header=None, names=['label','body_text'])
data.head()

# Remove punctuation
import string
string.punctuation

def remove_punct(text):
  return ''.join([char for char in text if char not in string.punctuation])

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))
data.head()

# Tokenization
import re

def tokenize(text):
  return re.split('\W+',text)

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))
data.head()

# Remove stopwords
import nltk
re.split('\s',"This is an interesting te+st")
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
  return [word for word in tokenized_list if word not in stopword]

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
data.head()
