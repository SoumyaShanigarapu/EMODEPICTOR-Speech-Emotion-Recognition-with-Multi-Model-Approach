#!/usr/bin/env python
# coding: utf-8

# In[15]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install resampy')


# In[ ]:


import librosa
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# In[ ]:


from IPython.display import Audio


# In[ ]:


# Specify the path to the file you want to check
file_path = '/content/drive/MyDrive/DA_Project/Audio_Files'

# Check if the file exists
if os.path.exists(file_path):
    print(f"The file {file_path} exists in your Google Drive.")
else:
    print(f"The file {file_path} does not exist in your Google Drive.")


# In[ ]:


audio_file_path = '/content/drive/MyDrive/DA_Project/03-01-08-01-02-01-24.wav'
Audio(audio_file_path)


# In[ ]:


get_ipython().system('pip install librosa')


# In[ ]:


import librosa
import librosa.display
import matplotlib.pyplot as plt


# In[ ]:


audio_file_path = '/content/drive/MyDrive/DA_Project/03-01-08-01-02-01-24.wav'
y, sr = librosa.load(audio_file_path)


# In[ ]:


plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()


# In[ ]:


import os
import pandas as pd

# Specify the path to your audio files
file_path = '/content/drive/MyDrive/DA_Project/Audio_Files'

file_emotion = []
file_path_list = []

# Iterate through each actor's directory
for actor_dir in os.listdir(file_path):
    actor_path = os.path.join(file_path, actor_dir)

    # Iterate through each file in the actor's directory
    for file in os.listdir(actor_path):
        # Split the file name to extract information
        part = file.split('.')[0].split('-')

        # The third part in each file represents the emotion associated with that file
        file_emotion.append(int(part[2]))

        # Create the full path to the file
        file_path_list.append(os.path.join(actor_path, file))

# Create DataFrames for emotion and file path
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path_list, columns=['Path'])

# Concatenate the DataFrames to create the final DataFrame
dataset_df = pd.concat([emotion_df, path_df], axis=1)

# Map integer emotions to actual emotions
emotion_mapping = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}
dataset_df['Emotions'] = dataset_df['Emotions'].map(emotion_mapping)

# Display the first few rows of the DataFrame
dataset_df.head()


# In[ ]:


import matplotlib.pyplot as plt

# Define colors for each emotion
emotion_colors = {
    'neutral': 'blue',
    'calm': 'green',
    'happy': 'yellow',
    'sad': 'purple',
    'angry': 'red',
    'fear': 'orange',
    'disgust': 'brown',
    'surprise': 'pink'
}

# Plot the count of each emotion with custom colors
plt.figure(figsize=(6, 6))
dataset_df['Emotions'].value_counts().sort_index().plot(kind='bar', color=[emotion_colors[emotion] for emotion in dataset_df['Emotions'].unique()])
plt.title('Distribution of Emotions in the Dataset')
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.show()


# In[14]:



# Define the path to your training data directory
TRAINING_FILES_PATH = '/content/drive/MyDrive/DA_Project/Audio_Files'

# Create an empty list to store (MFCCs, class label) pairs
data = []

for subdir, dirs, files in os.walk(TRAINING_FILES_PATH):
    for file in files:
        try:
            # Get MFCCs based on the sample_rate from the audio file
            X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

            # Extract the class label from the file name (assuming it's in the file name)
            file_class = int(file[7:8])  # Assumes class label is at position 7 in the filename

            # Create a tuple of (MFCCs, class label) and append to the data list
            arr = (mfccs, file_class)
            data.append(arr)

        except ValueError as err:
            print(err)
            continue

# Your 'data' list now contains pairs of MFCC features and their corresponding class labels.


# In[16]:


SAVE_DIR_PATH = 'DA_Project'


# In[17]:


import joblib


# In[18]:


# Extract MFCC features and labels for saving
x, y = zip(*data)
x = np.asarray(x)
y = np.asarray(y)

# Save MFCC features and labels to joblib files
if not os.path.isdir(SAVE_DIR_PATH):
    os.makedirs(SAVE_DIR_PATH)

joblib.dump(x, os.path.join(SAVE_DIR_PATH, "x.joblib"))
joblib.dump(y, os.path.join(SAVE_DIR_PATH, "y.joblib"))


# Training CNN MODEL

# In[19]:


import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the MFCC features and labels
x = joblib.load(os.path.join(SAVE_DIR_PATH, "x.joblib"))
y = joblib.load(os.path.join(SAVE_DIR_PATH, "y.joblib"))


# In[20]:



y_one_hot = to_categorical(y)


# In[21]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)


# In[22]:


# Reshape the input data for CNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


# In[23]:


from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# Build the CNN model
model = Sequential()
# Check the input shape of your training data
print(x_train.shape[1])



# Build the CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))  # Assuming you have 10 classes

# Print the model summary
model.summary()


# In[24]:



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[25]:



# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# In[26]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# In[27]:



# Save the model
model.save('audio_classification_model.h5')


# In[28]:


from tensorflow.keras.optimizers import Adam

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()


# In[29]:


import pandas as pd

# Assuming 'model' is your trained model and 'x_test' is your test data
y_pred = model.predict(x_test)

# Convert numerical class indices to emotion labels
predicted_labels = [emotion_mapping[i] for i in np.argmax(y_pred, axis=1)]
actual_labels = [emotion_mapping[i] for i in np.argmax(y_test, axis=1)]

# Create a DataFrame with predicted and actual labels
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = predicted_labels
df['Actual Labels'] = actual_labels

df.head(10)


# In[30]:


pip install tensorflow keras


# In[31]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'model' is your trained model and 'x_test' is your test data
y_pred = model.predict(x_test)

# Convert numerical class indices to emotion labels
predicted_labels = [emotion_mapping[i] for i in np.argmax(y_pred, axis=1)]
actual_labels = [emotion_mapping[i] for i in np.argmax(y_test, axis=1)]

# Create a confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=emotion_mapping.values(), columns=emotion_mapping.values())

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Emotional Labels)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


# In[32]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[33]:


# Assuming 'x' is your input data and 'y' is your labels
# Replace 'input_shape', 'num_classes', etc., with your actual values
input_shape = (x_train.shape[1], 1)  # Adjusted input shape for RNN
num_classes = 9  # Assuming you have 9 classes based on your code

#


# In[34]:



encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)


# In[35]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)


# In[36]:


# Build the RNN model
model_rnn = Sequential()
model_rnn.add(LSTM(128, input_shape=input_shape, activation='relu'))
model_rnn.add(Dropout(0.5))
model_rnn.add(Dense(64, activation='relu'))
model_rnn.add(Dropout(0.5))
model_rnn.add(Dense(num_classes, activation='softmax'))


# In[37]:



# Print the model summary
model_rnn.summary()


# In[38]:



# Compile the model
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[39]:


# Train the model
history_rnn = model_rnn.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))



# In[40]:


# Evaluate the model on the test set
loss_rnn, accuracy_rnn = model_rnn.evaluate(x_test, y_test)
print(f'Test Loss: {loss_rnn}, Test Accuracy: {accuracy_rnn}')


# In[41]:



# Save the model
model_rnn.save('audio_rnn_model.h5')


# In[42]:


from tensorflow.keras.optimizers import Adam

# Compile the RNN model
model_rnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the RNN model and store the training history
history_rnn = model_rnn.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plot the training and validation accuracy for RNN
plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs (RNN)')
plt.legend()
plt.show()


# In[43]:


# Assuming 'model_rnn' is your trained RNN model and 'x_test' is your test data
y_pred_rnn = model_rnn.predict(x_test)

# Convert numerical class indices to emotion labels
predicted_labels_rnn = [emotion_mapping[i] for i in np.argmax(y_pred_rnn, axis=1)]
actual_labels_rnn = [emotion_mapping.get(i, f'Unknown-{i}') for i in np.argmax(y_test, axis=1)]

# Get unique emotional labels from the dataset
emotional_labels = list(emotion_mapping.values())

# Create a confusion matrix
cm_rnn = confusion_matrix(actual_labels_rnn, predicted_labels_rnn, labels=emotional_labels)

# Create a DataFrame for better visualization
cm_df_rnn = pd.DataFrame(cm_rnn, index=emotional_labels, columns=emotional_labels)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df_rnn, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Emotional Labels - RNN)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()



# In[44]:


print("Unique values in y_test:", np.unique(np.argmax(y_test, axis=1)))


# In[45]:


import pandas as pd

# Assuming 'y_test' contains the true emotional labels and 'y_pred_rnn' contains predicted labels
actual_labels_rnn = [np.argmax(label) + 1 for label in y_test]
predicted_labels_rnn = [np.argmax(pred) + 1 for pred in y_pred_rnn]

# Emotion mapping
emotion_mapping = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}

# Map numerical indices to emotional labels
actual_labels_rnn = [emotion_mapping[label] for label in actual_labels_rnn]
predicted_labels_rnn = [emotion_mapping[label] for label in predicted_labels_rnn]

# Create a DataFrame for actual and predicted labels
labels_df_rnn = pd.DataFrame({
    'Actual Labels': actual_labels_rnn,
    'Predicted Labels': predicted_labels_rnn
})

# Display the first few rows of the DataFrame
labels_df_rnn.head()


# In[46]:


import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the MFCC features and labels
x = joblib.load(os.path.join(SAVE_DIR_PATH, "x.joblib"))
y = joblib.load(os.path.join(SAVE_DIR_PATH, "y.joblib"))


# In[47]:



# Flatten the MFCC features if needed (SVM expects 1D input)
x_flat = x.reshape(x.shape[0], -1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


# In[48]:


# Create and train the SVM model
svm_model = SVC(kernel='linear')  # You can experiment with different kernels (linear, rbf, etc.)
svm_model.fit(x_train, y_train)



# In[49]:


# Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Display classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[50]:


joblib.dump(svm_model, 'audio_classification_svm_model.joblib')


# In[ ]:





# In[51]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'x_test' contains your test data
# Make predictions on the test set
svm_predictions = svm_model.predict(x_test)

# Assuming 'y_test' contains the true emotional labels
# Map numerical indices to emotional labels
actual_labels_svm = [emotion_mapping[label] for label in y_test]

# Map numerical indices to emotional labels for predictions
predicted_labels_svm = [emotion_mapping[label] for label in svm_predictions]

# Get unique emotional labels from the dataset
emotional_labels = list(emotion_mapping.values())

# Create a confusion matrix
cm_svm = confusion_matrix(actual_labels_svm, predicted_labels_svm, labels=emotional_labels)

# Create a DataFrame for better visualization
cm_df_svm = pd.DataFrame(cm_svm, index=emotional_labels, columns=emotional_labels)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df_svm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Emotional Labels - SVM)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


# In[52]:


from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical

# Assuming 'y_test' is an array of integer labels
# Convert to one-hot encoding
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Compile the model with the appropriate configuration
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Make predictions on the test set
y_pred_cnn = model.predict(x_test)

# Convert one-hot encoded predictions back to class labels
predicted_labels_cnn = np.argmax(y_pred_cnn, axis=1)

# Calculate accuracy
accuracy_cnn = accuracy_score(y_test, predicted_labels_cnn)

print(f'CNN Model - Test Accuracy: {accuracy_cnn}')

# Calculate F1 score
f1_cnn = f1_score(y_test, predicted_labels_cnn, average='weighted')

print(f'CNN Model - F1 Score: {f1_cnn}')


# In[53]:


print("Shape of y_test:", y_test.shape)
print("Shape of y_test_one_hot:", y_test_one_hot.shape)


# In[54]:


from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical

# Assuming 'y_test' is an array of integer labels
# Convert to one-hot encoding
y_test_one_hot_rnn = to_categorical(y_test, num_classes=num_classes)

# Compile the RNN model with the appropriate configuration
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Make predictions on the test set for RNN
y_pred_rnn = model_rnn.predict(x_test)

# Convert one-hot encoded predictions back to class labels for RNN
predicted_labels_rnn = np.argmax(y_pred_rnn, axis=1)

# Calculate accuracy for RNN
accuracy_rnn = accuracy_score(y_test, predicted_labels_rnn)

print(f'RNN Model - Test Accuracy: {accuracy_rnn}')

# Calculate F1 score for RNN
f1_rnn = f1_score(y_test, predicted_labels_rnn, average='weighted')

print(f'RNN Model - F1 Score: {f1_rnn}')


# In[55]:


from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Assuming 'y_test' is an array of integer labels
encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)

# Make predictions on the test set for SVM
svm_predictions = svm_model.predict(x_test)

# Calculate accuracy for SVM
accuracy_svm = accuracy_score(y_test_encoded, svm_predictions)

print(f'SVM Model - Test Accuracy: {accuracy_svm}')

# Calculate F1 score for SVM
f1_svm = f1_score(y_test_encoded, svm_predictions, average='weighted')

print(f'SVM Model - F1 Score: {f1_svm}')

