# Classification Basics

#--- Preparing data for classification
with open('data/Course-Descriptions.txt') as t:
    descriptions = t.read().splitlines()

import nltk
from nltk.corpus import stopwords

# setup wordnet for lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer

# Custom tokenizer that will perform tokenization, stopword removal and lemmatization
def customtokenize(str):
    tokens = nltk.word_tokenize(str)
    nostop = list(filter(lambda token: token not in stopwords.words('english'), tokens))
    lemmatized = [lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized

# Generate TF-IDF matrix
vectorizer =  TfidfVectorizer(tokenizer=customtokenize)
tfidf=vectorizer.fit_transform(descriptions)

print('\Sample feature names identified: ',vectorizer.get_feature_names()[:25])
print('\nSize of TF-IDF matrix: ',tfidf.shape)
#--- Building the model - Naive Bayes classification
with open('data/Course-Classification.txt','r') as f:
    classifications = f.read().splitlines()

# Create labels and integer classes
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(classifications)
print('Classes found: ', le.classes_)

# Convert classes to integers for use with ML
int_classes = le.transform(classifications)
print('Classes converted to integers: ', int_classes)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Split as training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(tfidf, int_classes, random_state=0)
classifier = MultinomialNB().fit(xtrain, ytrain)

#--- Running Predictions
from sklearn import metrics

predictions = classifier.predict(xtest)
predictions

# Confusion Matrics
metrics.confusion_matrix(ytest, predictions)

# Prediction Accuracy
metrics.accuracy_score(ytest, predictions)


# Predict on entire corpus data
predictions=classifier.predict(tfidf)
# Confusion matrix
metrics.confusion_matrix(int_classes,predictions)
metrics.accuracy_score(int_classes, predictions)
