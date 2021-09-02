# How to implement word2vec

# Explore Pre-trained Embeddings
# - glove-twitter-{25/50/100/200}
# - glove-wiki-gigaword-{50/200/300}
# - word2vec-google-news-300
# - word2vec-ruscorpora-news-300

# !pip install -U gensim
import gensim.downloader as api

wiki_embeddings = api.load('glove-wiki-gigaword-100')

# Explore the word vector for "king"
wiki_embeddings['king']

# Find the words most similar to king based on the trained word vectors
wiki_embeddings.most_similar('king')

#--- Train the model
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth',100)

# Read in the data and clean up column names
messages = pd.read_csv('data/spam.csv', encoding='latin-1')
messages = messages.drop(labels = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
messages.columns = ['label', 'text']
messages.head()

# Clean data using the built in cleaner in gensim
messages['clean_text'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
messages.head()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(messages['clean_text'],messages['label'], test_size=0.2)

# Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train, size=100, window=5, min_count=2)

# Explore the word vector for 'king' based on our trained model
w2v_model.wv['king']

# Find the most similar words to "king" based on word vectors from our trained model
w2v_model.wv.most_similar('king')

# Generate a list of words the word2vec model learned word vectors for
w2v_model.wv.index2word

# Generate aggregated sentence vectors based on the word vectors for each word in the sentence
w2v_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in w2v_model.wv.index2word]) for ls in X_test])

# Why is the length of the sentence different than the length of the sentence vector?
for i, v in enumerate(w2v_vect):
        print(len(X_test.iloc[i]), len(v))

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
w2v_vect_avg = []

for vect in w2v_vect:
        if len(vect) != 0:
                w2v_vect_avg.append(vect.mean(axis=0))
        else:
                w2v_vect_avg.append(np.zeros(100))

# Are the sentence vector lengths consistent?
for i, v in enumerate(w2v_vect_avg):
        print(len(X_test.iloc[i]), len(v))
        
