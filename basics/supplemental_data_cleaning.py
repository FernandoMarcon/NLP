#--- Supplemental Data Cleaning
# Stemming and Lemmatizing
import re
import string
import nltk
import pandas as pd
pd.set_option('display.max_colwidth',100)

stopwords = nltk.corpus.stopwords.words('english')

data = pd.read_csv('data/SMSSpamCollection.tsv', sep = '\t', header=None, names=['label','body_text'])
data.head()

# Clean up text
def clean_text(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [word for word in tokens if word not in stopwords]
    return text
data['body_text_nostop'] = data['body_text'].apply(lambda x: clean_text(x.lower()))
data.head()

## Stemming
### PorterStemmer
ps = nltk.PorterStemmer()
[ps.stem(x) for x in ['grows','growing','grow']]

[ps.stem(x) for x in ['run','running','runner']]

# Stem text
def stemming(tokenized_text):
    return [ps.stem(word) for word in tokenized_text]
data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))
data.head()

# Lemmatizing (WordNet lemmatizer)
wn = nltk.WordNetLemmatizer()

ps.stem('meanness')
ps.stem('meaning')

wn.lemmatize('meanness')
wn.lemmatize('meaning')


ps.stem('goose')
ps.stem('geese')

wn.lemmatize('goose')
wn.lemmatize('geese')

def lemmatizing(tokenized_text):
    return [wn.lemmatize(x) for x in tokenized_text]

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))
data.head()
