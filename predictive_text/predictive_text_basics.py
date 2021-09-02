# Predictive Text basics
import os
import nltk

#--- Prepare data
base_file = open('data/Course-Descriptions.txt', 'rt')
raw_text = base_file.read()
base_file.close()

# Tokenization
token_list = nltk.word_tokenize(raw_text)

# Replace special characters
token_list2 = [word.replace("''","") for word in token_list]

# Remove punctuations
token_list3 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list2))

# Convert to lower case
token_list4 = [word.lower() for word in token_list3]

token_list4[:10]

#--- Building the ngrams DB
from nltk.util import ngrams

# Use a sqlite database to store ngrams information
import sqlite3
conn = sqlite3.connect(':memmory:')

# table to store first word, second word and count fo occurance
conn.execute("""DROP TABLE IF EXISTS NGRAMS""")
conn.execute("""CREATE TABLE NGRAMS
                (FIRST TEXT NOT NULL,
                SECOND TEXT NOT NULL,
                COUNTS INT NOT NULL,
                CONSTRAINT PK_GRAMS PRIMARY KEY (FIRST,SECOND));""")

# Generate bigrams
bigrams = ngrams(token_list4, 2)

# Store bigrams in DB
for i in bigrams:
    insert_str="""INSERT INTO NGRAMS (FIRST, SECOND, COUNTS)
                VALUES ('" + i[0] +"','" + i[1]+"',1)
                ON CONFLICT(FIRST,SECOND) DO UPDATE SET COUNTS=COUNTS + 1"""
    conn.execute(insert_str)

# Look at sample data from the table
cursor = conn.execute('SELECT FIRST, SECOND, COUNTS FROM NGRAMS LIMIT 5')
for gram_row in cursor:
    print('FIRST=',gram_row[0],'SECOND=',gram_row[1],'COUNT=',gram_row[2])

#--- Recommending the next word
# Function to query DB and find next word
def recommend(str):
    nextword = []
    # Find next words, sort them by most accurate
    cur_filter = conn.execute("SELECT SECOND FROM NGRAMS \
                                WHERE FIRST='"+ str + "' \
                                ORDER BY COUNTS DESC")

    # Build a list ordered from most frequente to least frequent next word
    for filt_row in cur_filter:
        nextword.append(filt_row[0])

    return nextword

# Recomemnd for words data and science
print('Next word for data ',recommend('data'))
print('\nNext word for science ', recommend('science'))
