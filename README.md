# Natural Language Processing

The core component of Natural Language Processing (NLP) is extracting information from human language.

the need to analyze and understand text data is growing everyday.
More and more data generated today is free text
- Web: blogs, comments, reviews, notes
- Social media: messages, hastags, references
- Operations: logs, trails
- Emails
- Voice transcriptions

Volume and lack of structure provide additional challenges to acquire, process, and analyze text.

> __*Document*__: A collection of sentences that represent a specific fact or entity.

> __*Copus*__: a collection of similar documents

## Main Application Topics
- Sentiment analysis
- Topic modeling
- Text classification
- Sentence segmentation or part-of-speech tagging
- ...

## General Pipeline
1. Raw text
2. Tokenize - tell the model what to look at
3. Clean text - remove stop words/ponctuation, stemming, etc.
4. Vectorize - convert to numeric form
5. Machine Learning algorithm - fit/train model
6. Model Selection

## Unstructured Data
Binary data, no delimiters, no indication of rows

## Cleansing Text
- Formatting and standardization (e.g., dates)
- Remove punctuation
- Remove abbreviations
- Case conversion
- Remove elements like hashtags

## Stop-word Removal
> A group of words that carry no meaning by themselvs (in, and, the which)
- Not required for analytics
- A standard or custom stop-words dictionary can be used.

## Stemming
> A stem is the base parte of the word, to which affixes can be attached for derivatives.

Stemming keeps only the base word, thus reducing the total words in the corpus.

Though they may have different affixes, words that share the same stem have similar semantic meaning. Stemming is able to determine that 'learned' and 'learning' , though they have different affixes, each contain the same root word 'learn'.
- Reduces the corpus of words the model is exposed to
- Explicitly correlates words with similar meanings

# Lemmatizing
> Similar to stemming, but produces a proper root word that belongs to the language

Uses a dictionary to match words to their root word

Process of grouping together the inflected forms of a word so they can be analyzed as a single term, identified by the word's lemma.
Using vocabulary analysis of words aiming to remove inflectional endings to return the dictionary form of a word

## Stemming vs. Lemmatizing
- To goal of both is to condense derived words into their base forms
    - Stemming is typically faster as it simply chops off the end of a word using heuristics, without any understanding of the context in which a word is used.
    - Lemmatizing is typically more accurate as it uses more informed analysis to create groups of words with similar meaning based on the context aroud the word.

## Parts-of_Speech (POS) Tagging
- POS tagging involves identifying the part of speech for each word in a corpus
- Used for entity recognition, filtering, and sentiment analysis
- Parts of speech tagging are used by chatbots to understand natural language and sentiments.

|Word|POS|Description|
--- | --- | ---
|Man|NN|Noun|
|Engage|VBP|Verb Singular Present|
|Top|JJ|Adjective|

## Vectorization
Raw text needs to be converted to numbers so that Python and the algorithms used for machine learning can understand.

> __*Vectorizing:*__ Process of encoding text as integers to create feature vectors.

> __*Feature vector:*__ An n-dimensional vector of numerical features that represent some object.

Vectorizers should be fit on the training set and only be used to transform the test set.

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
> Process that alters each data point in a certain column in a systematic way
- __Power transformations__ (e.g., $-x^2$, $???{x}$)
    - _Transformation Process_
        1. Determine what range of exponents to test.
        2. Apply each transformation to each value of your chosen feature.
        3. Use some criteria to determine which of the transformations yield the best distribution.
    - _Box-Cox Power Transformations_  (Base Form: $y^x$)
        |X|Base Form| Transformation|
        ---|---|---
        |-2|$y^-2$|$\frac{1}{y??}$|
        |-1|$y^-1$|$\frac{1}{y}$|
        |-0.5|$y^-1/2$|$\frac{1}{???y}$|
        |0|$y^0$|$log(y)$|
        |0.5|$y^1/2$|$???y$|
        |1|$y^1$|$y$|
        |2|$y^2$|$y??$|

- __Standardizing data__

## Machine Learning

> "The field of study that gives computers the ability to learn without being explicitly programmed." (Arthur Samuel, 1959)

> "A computer program is said to learn from experience E with respect to some task T and som performance measure P, if its performance on T, as measured by P, improves with experience E." (Tom Mitchell, 1998)

> "Algorithms that 'can figure out how to perform important tasks by generalizing from examples'" (University of Washington, 2012)

> "Practice of using algorithms to parse data, learn from it, and then make a determination or prediction about something in the world" (NVIDIA, 2016)

Two Broad Types of Machine Learning:
> _Supervised Learning_: Inferring a function from labeled training data to make predictions on unseen data

> _Unsupervised Learning_: Deriving structure from data where we don't know the effect of any of the variables


### Holdout Test Set
Sample of data not used in fitting a model for the purpuse of evaluating the model's ability to generalize unseen data

> **_K_-Fold Cross-Validation**: The full data set is divided into _k_-subsets and the holdout method is repeated _k_ times. Each time, one of the _k_-subsets is used as the test set and the other _k_-1 subsets are put together to be used to train the model.

### Evaluation Metrics
$$Accuracy = \frac{\#  predicted\ correctly}{total\ \#\ of\ observations}$$
$$Precision = \frac{\#\ predicted\ as\ spam\ that\ are\ actually\ spam}{total\ \#\ predicted\ as\ spam}$$
$$Recall = \frac{\#\ predicted\ as\ spam\ that\ are\ actually\ spam}{total\ \#\ that\ are\ actually\ spam}$$

### Ensemble Method
> Techinique that creates multiple models and then combines them to produce better results than any of the single models individually.

#### Random Forest
> Ensemble learning method that constructs a collection of decision tress and then aggregates the predictions of each tree to determine the final prediction
- Can be used for classification or regression
- Easily handles outliers, missing values, etc.
- Accepts various types of inputs (continuous, ordinal,etc.)
- Less likely to overfit
- Outputs feature importance

> Grid-search: Exhaustively search all paramenters combinations in a given grid to determine the best model

> Cross-validation: Divide a dataset into k subsets and repeat the holdout method k times where a different subset is used as the holdout set in each iteration.

#### Gradient Boosting
> Ensemble learning method that takes an iterative approach to combining wak learners to create a strong learner by focusing on mistakes of prior iterations

Trade-offs of Gradient Boosting
- _Pros_
    - Extremely powerful
    - Accepts various types of inputs
    - Can be used for classification or regression
    - Outputs feature importance
- _Cons_
    - Longer to train (can't parallize)
    - More likely to overfit
    - More difficult to properly tune

#### Random Forest vs. Gradient Boosting
Both are ensemble methods based on decision tress.
|Random Forest|Gradient Boosting|
--- | ---
|Bagging|Boosting|
|Training done in parallel|Training done iteratively|
|Unweighted voting for final prediction|Weighted voting for final prediction|
|Easier to tune, harder to overfit|Harder to tune, easier to overfit|

## Model Selection
### Process
1. Split the data into training and test set.
2. Train vectorizers on training set and use that to transform test set.
3. Fit best random forest model and best gradient boosting model on training set and predict on test set.
4. Thoroughly evaluate results of these two models to select best model

__Further evaluation__:

    - Slice test set
    - Examine text messages the model is getting wrong


__Results trade-off__: consider business context

    - Is predict time of 0.213 vs. 0.135 going to create a bottleneck?
    - Precision/recall
        + Spam filter - optimize for precision
        + Antivirus software - optimize for recall

## Embeddings
### word2vec
> is a shallow, two-layer neural network that accepts a text corpus as an input, and it returns a set of vectors (also known as embeddings); each vector is a numeric representation of a given word.

> "You shall know a word by the company it keeps."

- `gensim` package pre-trained Embeddings:
    - glove-twitter-{25/50/100/200}
    - glove-wiki-gigaword-{50/200/300}
    - word2vec-google-news-300
    - word2vec-ruscorpora-news-300

### doc2vec
> is a shallow, two-layer neural network that accepts a text corpus as an input, and it returns a set of vectors (aka embeddings); each vector is a numeric representation of a given sentence, paragraph, or document.

__Pre-trained Document Vectors__
    There are not as many options as there are for word vectors. There also is not an easy API to read these in like there is for `word2vec` so it is more time consuming.

    Pre-trained vectors from training on Wikipedia and Associated Press News can be found [here](https://github.com/jhlau/doc2vec).

## Recurrent Neural Network
> Pattern matching through the connection of many very simple functions to create one very powerful functino; __this functin has an understanding of the data's sequential nature (using feedback loops that form a sense of memory)__

## Best Practices
### Storing Text Data
- Use suitable free-format big-data storage for text (HDFS, S3, or Google Cloud Storage)
- Create indexes on key data elements for easy access (MongoDB, Elasticsearch)
- Store processd text like tokens and TF-IDF

### Processing text data
- Filter text as early as possible in the processing cycle
- Use an exhaustive and context-specific stop-word list
- Identify domain-specific data for special use
- Eliminate data with low frequency
- Build a clean and indexed corpus

### Scalable processing of text data
- Use technologies that allow parallel access and storage (Kafka, HDFS, MongoDB, and so on)
- Oricess eachg document independently with map() functions (in Hadoop or Apache Spark)
- Use reduce() functions late in the processing cycle

## Souces
- [NLP with Python for Machine Learning Essential Training (LinkedIn)](https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training/)
- [Advanced NLP with Python for Machine Learning (LinkedIn)](https://www.linkedin.com/learning/advanced-nlp-with-python-for-machine-learning/what-you-should-know)
- [ Processing Text with Python Essential Training (LinkedIn)](https://www.linkedin.com/learning/processing-text-with-python-essential-training/the-need-for-text-mining-skills-in-data-science)
- [ Text Analytics and Predictions with Python Essential Training (LinkedIn)](https://www.linkedin.com/learning/text-analytics-and-predictions-with-python-essential-training/the-need-for-text-mining-skills-in-data-science)
