# Evaluate Random Forest with GridSearchCV
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv('data/SMSSpamCollection.tsv', sep='\t', names=['label','body_text'])

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(' ')), 3)

def clean_text(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tolkens = re.split('\W+', text)
    text = [ps.stem(word) for word in tolkens if word not in stopwords]
    return text

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

# TF-IDF
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)

# CountVectorizer
count_vect = CountVectorizer(analyzer=clean_text)
X_count = count_vect.fit_transform(data['body_text'])
X_count_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_count.toarray())], axis=1)
X_count_feat.head()

#--- Exploring parameter settings using GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
param = {'n_estimators': [10,50,100],
        'max_depth': [10,20,30,None]}

gs = GridSearchCV(rf, param, cv=5, n_jobs=2)
gs_fit = gs.fit(X_tfidf_feat, data['label'])
gs_dfidf = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)
gs_dfidf.head()

gs = GridSearchCV(rf, param, cv=5, n_jobs=2)
gs_fit = gs.fit(X_count_feat, data['label'])
gs_count_feat = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)
gs_count_feat.head()
