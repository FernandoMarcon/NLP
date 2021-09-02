import os
import nltk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


# Read the file using standard python libraries
with open('data/Spark-Course-Description.txt', 'r') as fh:
    filedata = fh.read()

filedata[0:200]

#--- Reading using NLTK CorpusReader
# nltk.download('punkt')
# Read the file into a corpus. The same command can read an entire directory
corpus=PlaintextCorpusReader('data/','Spark-Course-Description.txt')
raw_text = corpus.raw()
print(raw_text)

#--- Exporting the Corpus
# Extract the file IDs from the corpus
print('Files in this corpus: ', corpus.fileids())

# Extract paragraphs from the corpus
paragraphs = corpus.paras()
print('\n Total paragraphs in this corpus: ', len(paragraphs))

# Extract sentences from the corpus
sentences = corpus.sents()
print('\n Total sentences in this corpus: ',len(sentences))
print('\n The first sentence: ', sentences[0])

# Extract words from the corpus
print('\ Words in this corpus: ', corpus.words())


#--- Analyze the Corpus
# The NLTK library provides a numkber of functions to analyze the distributions and aggregates for data in the corpus
# Find the freequency distribution of words in the corpus
course_freq_dist=nltk.FreqDist(corpus.words())

# Print most commonly used words
print('Top 10 words in the corpus: ', course_freq_dist.most_common(10))

# Find the distribution for a specific word
print('\n Distribution for \"Spark": ', course_freq_dist.get('Spark'))

#### Text Cleansing and Extraction ####
#--- Tokenization
# Extract tokens
token_list = nltk.word_tokenize(raw_text)
token_list[:20]

#--- Cleasing Text
# Remove punctuation
token_list2 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list))

# Convert to Lower Case
token_list3 = [word.lower() for word in token_list2]

#--- Stop word removal
# nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove stopwords
token_list4 = list(filter(lambda token: token not in stopwords.words('english'), token_list3 ))

#--- Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Stem data
token_list5 = [stemmer.stem(word) for word in token_list4]

#--- Lemmatization
# Use the wordnet library to map words to their lemmatized form
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
token_list6 = [lemmatizer.lemmatize(word) for word in token_list4]

# Comparison of tokens between raw, stemming and lemmatization
print('Raw: ', token_list4[20], ', Stemmed: ', token_list5[20], ', Lemmatized: ', token_list6[20])

#### Advanced Text Processing
#--- Building n-grams
from nltk.util import ngrams
bigrams = ngrams(token_list6, 2)
trigrams = ngrams(token_list6, 3)

#--- Tagging parts of speech
# nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(token_list4)[:10]

#--- TF-IDF
 from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Use a small corpus for each visualization
vector_corpus = [
    'NBA is a Basketball league',
    'Basketball is popular in America.',
    'TV in America telecast BasketBall.'
]

# Create a vectorizer for english language
vectorizer = TfidfVectorizer(stop_words='english')

# Create the vector
tfidf = vectorizer.fit_transform(vector_corpus)

# Tokens used as features are:
vectorizer.get_feature_names()

# Size of array. Each row represents a document. Each colum,n represents a feature/token
tfidf.shape

# Actual TF-IDF array
tfidf.toarray()
