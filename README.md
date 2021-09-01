The core component of Natural Language Processing (NLP) is extracting information from human language.

## General Pipeline
1. Raw text
2. Tokenize - tell the model what to look at
3. Clean text - remove stop words/ponctuation, stemming, etc.
4. Vectorize - convert to numeric form
5. Machine Learning algorithm - fit/train model

## Unstructured Data
Binary data, no delimiters, no indication of rows

## Stemming
Though they may have different affixes, words that share the same stem have similar semantic meaning. Stemming is able to determine that 'learned' and 'learning' , though they have different affixes, each contain the same root word 'learn'.
- Reduces the corpus of words the model is exposed to
- Explicitly correlates words with similar meanings

# Lemmatizing
Process of grouping together the inflected forms of a word so they can be analyzed as a single term, identified by the word's lemma.
Using vocabulary analysis of words aiming to remove inflectional endings to return the dictionary form of a word

## Stemming vs. Lemmatizing
- To goal of both is to condense derived words into their base forms
    - Stemming is typically faster as it simply chops off the end of a word using heuristics, without any understanding of the context in which a word is used.
    - Lemmatizing is typically more accurate as it uses more informed analysis to create groups of words with similar meaning based on the context aroud the word.

## Vectorization
Raw text needs to be converted to numbers so that Python and the algorithms used for machine learning can understand.

> __*Vectorizing:*__ Process of encoding text as integers to create feature vectors.

> __*Feature vector:*__ An n-dimensional vector of numerical features that represent some object.

### Types
- __Count vectorization__
- __N-grams__
    > Creates a document-term matrix where counts still occupy the cell but instead of the columns representing single terms, they represent all combinations of adjacent words of length n in your text.

    Ex: "NLP is an interesting topic"

    |n|Name|Tokens|
    --- | --- | ---
    |2|bigram|['NLP is','is an','an interesting','interesting topic']|
    |3|trigram|['NLP is an','is an interesting','an interesting topic']|
    |4|four-gram|['NLP is an interesting','is an interesting topic']|

- __TF-IDF__: Term frequency - inverse document frequency
    $$w_{i,f} = tf_{i,j} *log (\frac{N}{df_i})$$
    > $td_{i,j}$ = number of times $i$ occurs in $j$ divided by total number of terms in $j$
    > $df_i$ = number of documents containing $i$
    > $N$ = total number of documents

## Feature Engineering
> Creating new features or transforming your existing features to get the most out of your data.

### Creating New Features
- Length of text field
- Percentage of characters that are punctuation in the text
- Percentage of characters that are capitalized

#### Transformations
- Power transformations (square, square root, etc.)
- Standardizing data
- 
