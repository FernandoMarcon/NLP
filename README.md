The core component of Natural Language Processing (NLP) is extracting information from human language.

## General Pipeline
1. Raw text
2. Tokenize - tell the model what to look at
3. Clean text - remove stop words/ponctuation, stemming, etc.
4. Vectorize - convert to numeric form
5. Machine Learning algorithm - fit/train model
6. Model Selection

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
- __Power transformations__ (e.g., $-x^2$, $√{x}$)
    - _Transformation Process_
        1. Determine what range of exponents to test.
        2. Apply each transformation to each value of your chosen feature.
        3. Use some criteria to determine which of the transformations yield the best distribution.
    - _Box-Cox Power Transformations_  (Base Form: $y^x$)
        |X|Base Form| Transformation|
        ---|---|---
        |-2|$y^-2$|$\frac{1}{y²}$|
        |-1|$y^-1$|$\frac{1}{y}$|
        |-0.5|$y^-1/2$|$\frac{1}{√y}$|
        |0|$y^0$|$log(y)$|
        |0.5|$y^1/2$|$√y$|
        |1|$y^1$|$y$|
        |2|$y^2$|$y²$|

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

## Souces
- [NLP with Python for Machine Learning Essential Training  (LinkedIn)](https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training/)
