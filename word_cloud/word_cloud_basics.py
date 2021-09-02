# Word-Cloud Basics
!pip install wordcloud

with open("data/Course-Descriptions.txt",'r') as fh:
    filedata = fh.read()

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, max_words=25, background_color = 'white').generate(filedata)

#--- Displaying the Wordcloud
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#--- Enhancing the word cloud
# Add more words to ignore
stopwords.update(['many', 'using', 'want','value'])

# Redo stop words. Limit number of words
wordcloud = WordCloud(stopwords=stopwords, max_words=10, background_color='azure').generate(filedata)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
