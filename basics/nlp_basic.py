# NPL Basics
import nltk
import re
import pandas as pd
pd.set_option('display.max_colwidth',100)

#--- Read in semi-structured text data
data = pd.read_csv('data/SMSSpamCollection.tsv', sep = '\t', header=None)
data
