#### Implement doc2vec ####

import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)

# Read in data, clean it, and then split into train and test sets
messages = pd.read_csv('data/spam.csv', encoding='latin-1')
messages = messages.drop(labels = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
messages.columns = ['label','text']
messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
messages.head()

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'], messages['label'],test_size=.2)

# Create tagged document objects to prepare to train the model
tagged_docs = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)]

# Look at what a tagged document looks like
tagged_docs[0]

# Train a basic doc2vec model
d2v_model = gensim.models.Doc2Vec(tagged_docs, vector_size = 100, window=5, min_count =2)

d2v_model.infer_vector(['i','am','learning','nlp'])


 # Prepare these vectors to be used in a machine learning model
vectors = [[d2v_model.infer_vector(words)] for words in X_test]
vectors[0]
