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

## Vectorize
Is about converting to numeric form
