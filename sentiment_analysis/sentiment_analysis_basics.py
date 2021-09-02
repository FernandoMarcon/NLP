# Sentiment Analysis Basics
import os


with open('data/Movie-Reviews.txt','r') as fh:
    reviews = fh.readlines()

#--- Finding Sentiments by Review
from textblob import TextBlob

for review in reviews:
    sentiment = TextBlob(review)
    print('{:40} : {:10} : {:10}'.format(review[:40],sentiment.polarity,sentiment.subjectivity))

#--- Summarization and display
# Categorize Polarity into Positive, Negative, and Neutral
labels =['Negative','Neutral','Positive']
# Initialize count array
values = [0,0,0]

# Categorize each review
for review in reviews:
    sentiment = TextBlob(review)

    # Custom formula to convert polarity
    polarity = round((sentiment.polarity + 1) * 3) % 3

    # add the summary array
    values[polarity] = values[polarity] + 1

print('Final summarized counts: ', values)

import matplotlib.pyplot as plt
colors = ['Green','Blue','Red']

# Plot a pie chart
plt.pie(values, labels = labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=140)
plt.axis('equal')
plt.show()
