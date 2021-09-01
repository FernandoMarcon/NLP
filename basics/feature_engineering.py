# Feature Engineering

import pandas as pd
data = pd.read_csv('data/SMSSpamCollection.tsv', sep = '\t', names = ['label','body_text'])

#--- Feature Creation
# Create feature for text message length
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))
data.head()

# Create feature for % of text that is punctuation
import string

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(' ')), 3) * 100

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
data.head()

#--- Evaluate created features
from matplotlib import pyplot
import numpy as np
%matplotlib inline

bins = np.linspace(0, 200, 40)
pyplot.hist(data[data['label'] == 'spam']['body_len'], bins=bins, alpha=0.5,label='spam')
pyplot.hist(data[data['label'] == 'ham']['body_len'], bins=bins, alpha=0.5,label='ham')
pyplot.show()


pyplot.hist(data[data['label'] == 'spam']['punct%'], bins=bins, alpha=0.5,label='spam')
pyplot.hist(data[data['label'] == 'ham']['punct%'], bins=bins, alpha=0.5,label='ham')
pyplot.show()

# Plot the two new features
pyplot.hist(data['body_len'], bins)
pyplot.title('Body Length Distribution')
pyplot.show()


bins = np.linspace(0,50,40)
pyplot.hist(data['punct%'], bins)
pyplot.title('Punctuation % Distribution')
pyplot.show()


#--- Transformation
for i in [1,2,3,4,5]:
    pyplot.hist((data['punct%'])**(1/i), bins = 40)
    pyplot.title('Transformation: 1/{}'.format(str(i)))
    pyplot.show()
