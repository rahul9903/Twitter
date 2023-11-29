import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import matplotlib.pyplot as plt

# Read the data from CSV file
data = pd.read_csv('C:\\Users\\rahul\\Downloads\\spam.csv', encoding='latin-1')

# Preprocess the data
X = data['text']
y = data['label']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

# Define the encoder architecture
embedding_dim = 100
hidden_units = 256

encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(len(tokenizer.word_index) + 1, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units)(encoder_embedding)
encoder_outputs = Dense(hidden_units)(encoder_lstm)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)

# Define the decoder architecture
decoder_inputs = Input(shape=(hidden_units,))
decoder_dense = Dense(1, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_inputs)

decoder_model = Model(inputs=decoder_inputs, outputs=decoder_outputs)

# Compile the models
encoder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
decoder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the encoder model
encoder_history = encoder_model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=2, batch_size=32)

# Plot training and validation accuracy for the encoder
plt.figure(figsize=(8, 6))
plt.plot(encoder_history.history['accuracy'], label='Training Accuracy')
plt.plot(encoder_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Encoder Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss for the encoder
plt.figure(figsize=(8, 6))
plt.plot(encoder_history.history['loss'], label='Training Loss')
plt.plot(encoder_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Encoder Training and Validation Loss')
plt.legend()
plt.show()

# Train the decoder model
decoder_history = decoder_model.fit(encoder_model.predict(X_train_padded), y_train, validation_data=(encoder_model.predict(X_test_padded), y_test), epochs=2, batch_size=32)

# Plot training and validation accuracy for the decoder
plt.figure(figsize=(8, 6))
plt.plot(decoder_history.history['accuracy'], label='Training Accuracy')
plt.plot(decoder_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Decoder Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss for the decoder
plt.figure(figsize=(8, 6))
plt.plot(decoder_history.history['loss'], label='Training Loss')
plt.plot(decoder_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Decoder Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the models
encoder_loss, encoder_accuracy = encoder_model.evaluate(X_test_padded, y_test, verbose=0)
decoder_loss, decoder_accuracy = decoder_model.evaluate(encoder_model.predict(X_test_padded), y_test, verbose=0)

print("Encoder Test Loss:", encoder_loss)
print("Encoder Test Accuracy:", encoder_accuracy)
print("Decoder Test Loss:", decoder_loss)
print("Decoder Test Accuracy:", decoder_accuracy)
