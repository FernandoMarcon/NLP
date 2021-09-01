import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Read and Clean Data
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv('data/SMSSpamCollection.tsv', sep='\t', names=['label','body_text'])

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(' ')), 3) * 100

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
X_features.head()

#--- Random Forest through Holdout Test Set
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

features_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True),columns=['importance','features'])
features_importance.head()

y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average='binary')

print('''Recall: {}% of all spam that has come into your email was properly placed in the spam folder.
Precision: {}% of the mails in the spam folder is actually spam.
Accuracy: {}% of emails that have come into your email were correctly identified as spam or ham.
'''.format(round(recall * 100, 2),
            round(precision * 100, 3),
            round((y_pred == y_test).sum() / len(y_pred) * 100, 2)))
