import pandas as pd
import re
import nltk
import string
pd.set_option('display.max_colwidth',100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv('data/SMSSpamCollection.tsv', sep = '\t', names = ['label','body_text'])

def clean_text(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

# Apply CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
X_counts.shape
count_vect.get_feature_names()

# Apply CountVectorizer to smaller sample
data_sample = data[0:20]
count_vect_sample = CountVectorizer(analyzer=clean_text)
X_counts_sample = count_vect_sample.fit_transform(data_sample['body_text'])
X_counts_sample.shape
count_vect_sample.get_feature_names()

X_counts_df = pd.DataFrame(X_counts_sample.toarray(),
                            columns=count_vect_sample.get_feature_names())
X_counts_df.head()


#--- N-Grams
def clean_text2(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = ' '.join([ps.stem(word) for word in tokens if word not in stopwords])
    return text

data['cleaned_text'] = data['body_text'].apply(lambda x: clean_text2(x))
data.head()

ngram_vect = CountVectorizer(ngram_range=(2,2))
X_counts = ngram_vect.fit_transform(data['cleaned_text'])
X_counts.shape
ngram_vect.get_feature_names()

data_sample = data[0:20]
ngram_vect_sample = CountVectorizer(ngram_range=(2,2))
X_counts_sample = ngram_vect_sample.fit_transform(data_sample['cleaned_text'])
X_counts_sample.shape
ngram_vect_sample.get_feature_names()
X_counts_df = pd.DataFrame(X_counts_sample.toarray(),columns = ngram_vect_sample.get_feature_names())
X_counts_df.head()

#--- ID-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf.shape
tfidf_vect.get_feature_names()

data_sample = data[0:20]
tfidf_vect_sample = TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample = tfidf_vect_sample.fit_transform(data_sample['body_text'])
X_tfidf_sample.shape
tfidf_vect_sample.get_feature_names()
X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray(), columns = tfidf_vect_sample.get_feature_names())
X_tfidf_df.head()
