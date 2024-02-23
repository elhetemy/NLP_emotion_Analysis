
# Loading important Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional,SimpleRNN
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Loading dataset

column_names = ['text', 'label']
df= pd.read_csv('data.txt', delimiter=';',names=column_names)

df.head(4)

# data preprocessing and cleaning

df.info()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.info()

print(df['label'].value_counts())

## labels valuecount plot
import seaborn as sns
df.groupby('label').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()
articles = df['text'].values
labels = df['label'].values

len(articles),len(labels)

### Removing stop words and Lemmatization

# removing stop words and normalizing words to its base using Lemmatizer
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def process_sentence(Articles):
  clean_articles = []
  for article in Articles:
    clean_text = article.split()
    clean_text = [word.lower() for word in clean_text if word.lower() not in stop_words]
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text]
    clean_articles.append(clean_text)

  return clean_articles

cleaned_articles=process_sentence(articles)

# splitting data

from sklearn.model_selection import train_test_split

# split data into train and validation using sklearn
from sklearn.model_selection import train_test_split
train_articles, validation_articles, train_labels, validation_labels = train_test_split(cleaned_articles, labels, test_size=0.2, random_state=42)

print('train_labels', len(train_labels))
print('validation_articles', len(validation_articles))
print('validation_labels', len(validation_labels))

#Tokenization and Padding

vocab_size = 5000
embedding_dim = 64
max_length = 200
oov_tok = '<OOV>' #  Out of Vocabulary
training_portion = 0.8

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
len(train_sequences)

train_padded = pad_sequences(train_sequences, maxlen=max_length)

len(train_sequences[10]),len(train_padded[10])

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

label_tokenizer.word_index


# loading test data

test_df=pd.read_csv('test.txt',delimiter=';',names=column_names)

test_articles=test_df['text'].values
test_label=test_df['label'].values



#Applying LSTM Algorithm

tf.keras.backend.clear_session()

# LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(vocab_size,embedding_dim ))
lstm_model.add(Dropout(0.5))
lstm_model.add(Bidirectional(LSTM(embedding_dim)))
lstm_model.add(Dense(10,activation='softmax' ))
lstm_model.add(Dense(7,activation='softmax' ))

lstm_model.summary()

lstm_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 20
history = lstm_model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq),
                    verbose=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#LSTM is better to be used as it is not overfitted on the data

# testing the LSTM model using test data

cleaned_test_articles=process_sentence(test_articles)

cleaned_test_articles[140],test_label[140]

labels=['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
cleaned_test_articles[59],test_label[59]

txt=[cleaned_test_articles[59]]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = lstm_model.predict(padded)

print(pred)
print(np.argmax(pred))
print(labels[np.argmax(pred)-1])

# Save the tokenizer to a file

def save_tokenizer(tokenizer, filename='tokenizer.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)
        
# saving the model using pickle
pickle.dump(lstm_model, open('lstm_model.pkl', 'wb'))

save_tokenizer(tokenizer)
