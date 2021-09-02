#### Implementation of a Basic RNN ####
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth',100)

#--- Read In, Clean, and Split the Data
messages = pd.read_csv('data/spam.csv',encoding='latin-1')
messages = messages.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
messages.columns = ['label', 'text']
messages.head()
labels = np.where(messages['label']=='spam',1,0)

X_train, X_test, y_train, y_test = train_test_split(messages['text'], labels, test_size=0.2)

#--- Prep Data for Modeling
# install keras
# !pip install -U keras
# !pip install tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Initialize and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Use that tokenizer to transform the text messages in the training and test sets
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# What do these sequences look like?
X_train_seq[0]

# Pad the sequences so each sequence is the same length
X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)

# What do these padded sequences look like?
X_train_seq_padded[0]

#--- Build Model
import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Construct a simple RNN model
model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1,32))
model.add(LSTM(32, dropout=0, recurrent_dropout=0))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision_m, recall_m])

# Fit the RNN model
history = model.fit(X_train_seq_padded, y_train, batch_size=32, epochs=10, validation_data = (X_test_seq_padded, y_test))

# Plot the evaluation metrics by each epoch for the model to see if we are over or underfitting
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=20,20

for i in ['accuracy','precision_m','recall_m']:
    acc = history.history[i]
    val_acc = history.history['val_{}'.format(i)]
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label = 'Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Results for {}'.format(i))
    plt.legend()
    plt.show()
