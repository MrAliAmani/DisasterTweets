# -*- coding: utf-8 -*-
"""tf_nlp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GVfecLBOtxwAdSG3niFzVZBXR-3v6irY

# NLP, seq2seq
* one to many
* one to one
* many to one
* many to many
* many to many ()
"""

!nvidia-smi -L

!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from helper_functions import plot_loss_curves, compare_historys, unzip_data, create_tensorboard_callback

"""## get the dataset kaggle get started with NLP"""

!wget https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip
unzip_data('nlp_getting_started.zip')

"""## visualizing the data 
* tf Load test
* pandas (needs a lot of memory)
* python
"""

import pandas as pd
train_df, test_df = pd.read_csv('train.csv'), pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')
train_df, test_df, sample_submission_df

# 1 for disaster
train_df['text'][1]

# shuffle if the data is not sequential
train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.head()

# tf imbalanced classification
train_df_shuffled.target.value_counts()

len(train_df), len(test_df)

import random
random_index = random.randint(0, len(train_df) - 5)
for row in train_df_shuffled[['text', 'target']][random_index: random_index + 5].itertuples():
  _, text, target = row
  print(f'target: {target}', '(real disaster)' if target > 0 else '(not a real disaster)')
  print(f'Text: \n{text}\n')
  print('___\n')

"""## split the data """

from sklearn.model_selection import train_test_split
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled['text'].to_numpy(), 
                                                                            train_df_shuffled['target'].to_numpy(), 
                                                                            test_size=.1,
                                                                            random_state=42)
len(train_sentences), len(val_sentences), len(train_labels), len(val_labels)

train_sentences[:10], train_labels[:10]

"""## tokenization + embedding

### text vectorization (tokenization)
"""

from tensorflow.keras.layers import TextVectorization
text_vectorier = TextVectorization(max_tokens=None, # <OOV>
                                   standardize='lower_and_strip_punctuation',
                                   split='whitespace',
                                   ngrams=None, 
                                   output_mode='int',
                                   output_sequence_length=None)

# average number of tokens in training tweets
round(sum([len(i.split()) for i in train_sentences]) / len(train_sentences))

# set the text vectorization variables
max_vocab_length = 10000
max_length = 15         # how many of words of the tweet will our model see
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode='int',
                                    output_sequence_length=max_length)

text_vectorizer.adapt(train_sentences)

train_sentences[:10]

# create a sample text and tokenizer
sample_sentence = 'There is a flood out there.'
text_vectorizer([sample_sentence])

# random sentence from train_sentences
random_sentence = random.choice(train_sentences)
print(f'Original text:\n {random_sentence}\
      \n\nVectorized version:\n')
text_vectorizer([random_sentence])

words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5]
bottom_5_words = words_in_vocab[-5:]
print(f'Number of words in vocab: {len(words_in_vocab)}')
print(f'5 most common words: {top_5_words}')
print(f'5 least common words: {bottom_5_words}')

"""## Embedding layer"""

from tensorflow.keras import layers
embedding = layers.Embedding(input_dim=max_vocab_length,
                            output_dim=128,
                            input_length=max_length,
                            embeddings_initializer='uniform')
embedding

random_sentence = random.choice(train_sentences)
print(f'Original text:\n {random_sentence}\
      \n\nEmbedded version:\n')
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed

# single token embedding
sample_embed[0][0], sample_embed[0][0].shape, random_sentence, random_sentence[0]

"""## modeling a text dataset and experiment

## baseline

* model_0 naive bayes 
* model_1 ffn 
* model_2 lstm
* model_3 gru
* model_4 bi-lstm
* model_5 1d conv
* model_6 tensorflow_hub transfer learning (feature extractor)
* model_7 transfer learning on 10% data

* **steps**
* create a model
* build the model
* fit the model
* evaluate
"""

# model_0 base line
# multinomial naive bayes with tf-idf formula
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

model_0 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

model_0.fit(train_sentences, train_labels)

# evaluate the baseline model
baseline_score = model_0.score(val_sentences, val_labels)
print(f'Our baseline model achieved accuracy of {baseline_score*100:.2f}')

train_df['target'].value_counts()

baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]

from sklearn.metrics import confusion_matrix, classification_report
classification_report_dict = classification_report(val_labels, baseline_preds, output_dict=True)
classification_report_dict

!pip install tensorflow-addons
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
  model_results = {'accuracy':model_accuracy,
                   'precision':model_precision,
                   'recall':model_recall,
                   'f1':model_f1_score}
  return model_results

baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
baseline_results

from tensorflow_addons.metrics import F1Score
from helper_functions import calculate_results, create_tensorboard_callback
# tensorboard callback
SAVE_DIR = 'model_logs'

"""## model_1 FFN"""

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
 # solve the shape error with GlobalAveragePooling1D or Flatten or GlobalMaxPool1D layers
x = layers.GlobalAveragePooling1D(name='gobal_average_pooling_1d')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_1 = tf.keras.Model(inputs, outputs, name='model_1_dense')
model_1.summary()

embedding, text_vectorizer

# compile
model_1.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

train_labels.shape, train_sentences.shape, train_labels.dtype, train_labels.ndim,

model_1_history = model_1.fit(x=train_sentences,
                              y=train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                     experiment_name='model_1_dense')])

model_1_results = model_1.evaluate(val_sentences, val_labels)
model_1_results

plot_loss_curves(model_1_history)

model_1_pred_probs = model_1.predict(val_sentences)
model_1_pred_probs[:10], model_1_pred_probs.shape

# convert model_pred_probs to model_preds
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:10], model_1_preds.shape

model_1_results = calculate_results(y_true=val_labels,
                                    y_pred=model_1_preds)
model_1_results

from scipy.optimize.zeros import results_c
import numpy as np

np.array(list(model_1_results.values())) > np.array(list(baseline_results.values()))

"""###visualizing learned embeddings"""

# get the voacabulary
words_in_vocab = text_vectorizer.get_vocabulary()
len(words_in_vocab), words_in_vocab[:10]

model_1.summary()

# weight matrix of embdding layer
embed_weights = model_1.get_layer('embedding').get_weights()[0]
embed_weights

print(embed_weights.shape)

"""* projector.tensorflow.org"""

# create embedding files
import io

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(words_in_vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = embed_weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

# dowbload from colab to upload to projector
try:
  from google.colab import files
  files.download('vectors.tsv')
  files.download('metadata.tsv')
except Exception:
  pass

"""## model_2 LSTM"""

train_sentences[:10]

# lstm model
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype='string')
x = text_vectorizer(inputs)
x = embedding(x)
# print(x.shape)
# x = layers.LSTM(units=64, return_sequences=True)(x)
# print(x.shape)
x = layers.LSTM(64)(x)
# print(x.shape)
# x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_lstm")
model_2.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])
model_2.summary()

model_2, train_sentences, train_labels

from keras.saving.legacy import saved_model
model_2_history = model_2.fit(train_sentences, 
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                     experiment_name='model_2_lstm')])

plot_loss_curves(model_2_history)

model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs[:10]

model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]

model_2_results = calculate_results(y_true=val_labels,
                                    y_pred=model_2_preds)
model_2_results

np.array(list(model_2_results)) > np.array(list(model_1_results))

np.array(list(model_2_results)) > np.array(list(baseline_results))

"""## model_3 GRU model"""

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
# x = layers.GRU(64, return_sequences=True)(x)
# x = layers.LSTM(64, return_sequences=True)(x)
x = layers.GRU(64)(x)
# x = layers.GlobalAveragePooling1D(name='global_aaverage_pooling_1d')(x)
# x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_3 = tf.keras.Model(inputs, outputs, name='model_3_gru')

model_3.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_3.summary()

model_3_history = model_3.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                     experiment_name='model_3_gru')])

model_3_pred_probs = model_3.predict(val_sentences)
model_3_pred_probs[:10]

model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]

plot_loss_curves(model_3_history)

model_3_results = calculate_results(y_true=val_labels,
                                    y_pred=model_3_preds)
model_3_results

np.array(list(model_3_results)) > np.array(list(baseline_results))

"""## model_4_bidirectional lstm """

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64))(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_4 = tf.keras.Model(inputs, outputs, name='model_3_bidirectional')

model_4.compile(loss='binary_crossentropy', 
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_4.summary()

model_4_history = model_4.fit(train_sentences, 
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                    experiment_name='model_3_bidirectional')])

model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]

model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]

model_4_results = calculate_results(y_true=val_labels,
                                    y_pred=model_4_preds)
model_4_results

np.array(list(model_4_results)) > np.array(list(baseline_results))

plot_loss_curves(model_4_history)

"""## model_5_conv1d"""

embedding_test = embedding(text_vectorizer(['This is a test sentence!']))
conv_1d = layers.Conv1D(filters=32,
                        kernel_size=5,
                        strides=1,
                        activation='relu',
                        padding='valid')
conv_1d_output = conv_1d(embedding_test)
max_pool = layers.GlobalMaxPool1D()
max_pool_output = max_pool(conv_1d_output)
embedding_test.shape, conv_1d_output.shape, max_pool_output.shape

embedding_test

conv_1d_output

max_pool_output

inputs = layers.Input(shape=(1,), dtype='string')
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='valid')(x)
x = layers.GlobalMaxPool1D(name='global_max_pool_1d')(x)
# x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_5 = tf.keras.Model(inputs, outputs, name='model_5_conv1d')

model_5.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_5.summary()

model_5_history = model_5.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                    experiment_name='model_5_conv1d')])

model_5_pred_probs = model_5.predict(val_sentences)
model_5_pred_probs[:10]

model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
model_5_preds[:10]

model_5_results = calculate_results(y_true=val_labels,
                                    y_pred=model_5_preds)
model_5_results

np.array(list(model_5_results)) > np.array(list(baseline_results))

plot_loss_curves(model_5_history)

"""## model_6 transfer learning pretrainerd sentence encoder (USE)"""

import tensorflow_hub as hub

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
embed_samples = embed([random_sentence])
embed_samples[0, :50], embed_samples.shape

embed_samples, embed_samples.shape

sentence_encoder_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name='USE')

model_6 = models.Sequential([
    sentence_encoder_layer,
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='model_6_USE')

model_6.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_6.summary()

model_6_history = model_6.fit(train_sentences,
                             train_labels,
                             epochs=5,
                             validation_data=(val_sentences, val_labels),
                             callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                    experiment_name='model_6_USE')])

model_6_pred_probs = model_6.predict(val_sentences)
model_6_pred_probs[:10]

model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_preds[:10]

model_6_results = calculate_results(y_true=val_labels,
                                    y_pred=model_6_preds)
model_6_results

np.array(list(model_6_results)) > np.array(list(baseline_results))

plot_loss_curves(model_6_history)

"""## model_7 tf hub pretrained USE with 10% data
* not samping from train-df_shuffled because information leakage occurs when validation data is present in the sampled training data
"""

train_df_shuffled.head(), train_df_shuffled['target'].value_counts()

train_10_percent_split = int(.1 * len(train_sentences))
train_sentences_10_percent = train_sentences[:train_10_percent_split]
train_labels_10_percent = train_labels[:train_10_percent_split]
len(train_sentences_10_percent), len(train_labels_10_percent)

pd.Series(np.array(train_labels_10_percent)).value_counts()

model_7 = tf.keras.models.clone_model(model_6)

model_7.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_7.summary()

model_7_history = model_7.fit(train_sentences_10_percent,
                              train_labels_10_percent,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                       experiment_name='tfhub_sentence_encoder_10_percent')])

model_7_pred_probs = model_7.predict(val_sentences)
model_7_pred_probs[:10]

model_7_preds = tf.squeeze(tf.round(model_7_pred_probs))
model_7_preds[:10]

model_7_results = calculate_results(y_true=val_labels,
                                    y_pred=model_7_preds)
model_7_results

plot_loss_curves(model_7_history)

np.array(list(model_7_results)) > np.array(list(baseline_results))

"""# compare the models"""

all_model_results = pd.DataFrame(
    {'0-baseline':baseline_results,
     '1-FNN':model_1_results,
     '2-LSTM':model_2_results,
     '3-GRU':model_3_results,
     '4-Bidirectional':model_4_results,
     '5-Conv1D':model_5_results,
     '6-USE':model_6_results,
     '7-USE_10_percent_data':model_7_results}
)
all_model_results = all_model_results.transpose()

plot_loss_curves(model_7_history)

# all_model_results['accuracy'] /= 100
# all_model_results

# plot and compare all of the model results
all_model_results.plot(kind='bar', figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))

# sort f1 score
all_model_results.sort_values('f1', ascending=False)['f1'].plot(kind='bar', figsize=(10, 7))

!tensorboard dev upload --logdir ./model_logs/ \
--name "kaggle_introduction_to_nlp_models" \
--description "Different experiments and different models on kaggle introduction to nlp dataset with different sizes" \
--one_shot

"""* Done. View your TensorBoard at https://tensorboard.dev/experiment/Qka29ePnQW2a4mzhVW3cOw/"""

!tensorboard dev list

# !tensorboard dev delete --experiment_id

"""# save the model"""

model_6.save('models/tfhub_USE_model_all_data.h5')

model = models.load_model('models/tfhub_USE_model_all_data.h5',
                          custom_objects={'KerasLayer':hub.KerasLayer})

model.summary()

model_results = model.evaluate(val_sentences, val_labels)
model_results

model_6_results

model_6.save('models/tfhub_USE_model_all_data')

model = models.load_model('models/tfhub_USE_model_all_data')
model.summary()

model_results = model.evaluate(val_sentences, val_labels)
model_results

pred_probs = model.predict(val_sentences)
pred_probs[:10]

preds = tf.squeeze(tf.round(pred_probs))
preds[:10]

"""# most wrong prediction"""

!wget https://storage.googleapis.com/ztm_tf_course/08_model_6_USE_feature_extractor.zip
!unzip "08_model_6_USE_feature_extractor.zip"

model_pretrained = tf.keras.models.load_model('/content/08_model_6_USE_feature_extractor')

model_pretrained.evaluate(val_sentences, val_labels)

model_pretrained_pred_probs = model_pretrained.predict(val_sentences)
model_pretrained_pred_probs[:10]

model_pretrained_preds = tf.squeeze(tf.round(model_pretrained_pred_probs))
model_pretrained_preds[:10]

val_sentences.shape, val_labels.shape, model_pretrained_pred_probs.shape

val_df = pd.DataFrame({
    'text':val_sentences,
    'target':val_labels,
    'pred':model_pretrained_preds,
    "pred_prob":tf.squeeze(model_pretrained_pred_probs),
})
val_df.head()

most_wrong = val_df[val_df['target'] != val_df['pred']].sort_values('pred_prob', ascending=False)
most_wrong.head(10) # FP

most_wrong.tail() # FN

for row in most_wrong[:10].itertuples():
  _, text, target, pred, pred_prob = row
  print(f'Target: {target}, pred: {pred}, prob: {pred_prob:.2f}\n')
  print(f'Text: \n{text}\n')
  print('----\n\n')

for row in most_wrong[-10:].itertuples():
  _, text, target, pred, pred_prob = row
  print(f'Target: {target}, pred: {pred}, prob: {pred_prob:.2f}\n')
  print(f'Text: \n{text}\n')
  print('----\n\n')

"""# predict and visualize test data"""

test_df

test_sentences = test_df['text'].to_list()
len(test_sentences), test_sentences[:10]

test_samples = random.sample(test_sentences, 10)
for test_sample in test_samples:
  pred_prob = tf.squeeze(model_pretrained.predict([test_sample]))
  pred = tf.round(pred_prob)
  print(f'Pred: {pred}, Prob: {pred_prob}\n')
  print(f'Text: \n{test_sample}\n')
  print("----\n\n")

"""# twitter"""

tweets = ["""Fun fact of the day: http://zerotomastery.io gets well over 1,000,000+ visitors a month. Jump on the bandwagon while it's still early. It won't be a "well kept secret of the few" for very much longer...""", 
          """The station master involved in Greece’s worst-ever train disaster will appear in court on Saturday, as mass protests broke out over the crash that killed at least 57 people https://aje.io/515ygl"""]
tweet_labels = [0, 1]

tweet_preds = give_predictions(model, tweets)
tweet_preds

"""# the speed/scoer tradeoff"""

model_6_results, baseline_results

import time

# time taken for model to predict
def pred_timer(model, samples):
  """
  Times how long a model takes to make predictions on samples
  """
  start_time = time.perf_counter()
  model.predict(samples)
  end_time = time.perf_counter()
  total_time = end_time - start_time
  time_per_pred = total_time / len(samples)
  return total_time, time_per_pred

model_6_total_time, model_6_time_per_pred = pred_timer(model=model_pretrained, 
                                                            samples=val_sentences)
model_6_total_time, model_6_time_per_pred

baseline_total_time, baseline_time_per_pred = pred_timer(model=model_0,
                                                        samples=val_sentences)
baseline_total_time, baseline_time_per_pred

model_pretrained_results = calculate_results(y_true=val_labels,
                  y_pred=model_pretrained_preds)
model_pretrained_results

import matplotlib.pyplot as plt

# plot the f1_score and time_per_pred
plt.figure(figsize=(10, 7))
plt.scatter(baseline_time_per_pred, baseline_results['f1'], label='baseline')
plt.scatter(model_6_time_per_pred, model_pretrained_results['f1'], label='model_6_pretrained')
plt.legend()
plt.title('F1-Score vs Time per Prediction')
plt.xlabel('Time per Prediction')
plt.ylabel('F1_score')

"""# model_8 sequential api FFN"""

model_8 = models.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation='sigmoid')
])

model_8.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_8.summary()

model_8_seq_history = model_8.fit(train_sentences, 
            train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name='model_8_fnn_sequential')])

model_8_pred_probs = model_8.predict(val_sentences)
model_8_preds = tf.squeeze(tf.round(model_8_pred_probs))
model_8_pred_probs[:10], model_8_preds[:10]

model_8_results = calculate_results(y_true=val_labels,
                                    y_pred=model_8_preds)
all_model_results = all_model_results.transpose()
all_model_results['8-fnn_seq'] = model_8_results.values()
all_model_results

"""# mode_9 lstm sequential"""

model_9 = models.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model_9.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

model_9.summary()

model_9_lstm_seq_history = model_9.fit(train_sentences,
                                       train_labels,
                                       epochs=5,
                                       validation_data=(val_sentences, val_labels),
                                       callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                              experiment_name='model_9_lstm_sequential')])

model_9_pred_probs = model_9.predict(val_sentences)
model_9_preds =tf.squeeze(tf.round(model_9_pred_probs))
model_9_pred_probs[:10], model_9_preds[:10]

model_9_results = calculate_results(y_true=val_labels,
                                    y_pred=model_9_preds)
all_model_results['9-lstm_seq'] = model_9_results.values()
all_model_results

"""# model_10 conv1d seq"""

model_10 = models.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation='sigmoid')
])

model_10.compile(loss='binary_crossentropy',
                 optimizer=optimizers.Adam(),
                 metrics=['accuracy'])

model_10.summary()

model_10_conv1d_history = model_10.fit(train_sentences, 
                                       train_labels,
                                       epochs=5,
                                       validation_data=(val_sentences, val_labels),
                                       callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                              experiment_name='model_10_conv1d_sequential')])

model_10_pred_probs = model_10.predict(val_sentences)
model_10_preds = tf.squeeze(tf.round(model_10_pred_probs))
model_10_pred_probs[:10], model_10_preds[:10]

model_10_results = calculate_results(y_true=val_labels, 
                                     y_pred=model_10_preds)
all_model_results['10-conv1d_seq'] = model_10_results.values()
all_model_results

"""# model_11 retrain on 10% data"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

model_11 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

model_11.fit(train_sentences_10_percent,
            train_labels_10_percent)

baseline_preds_10_percent = model_11.predict(val_sentences)
baseline_preds_10_percent[:10]

baseline_results_10_percent = calculate_results(y_true=val_labels,
                                                 y_pred=baseline_preds_10_percent)
baseline_results_10_percent

import numpy as np
np.array(list(baseline_results_10_percent)) > np.array(list(model_7_results))

all_model_results

all_model_results['11-nb_10_percent'] = baseline_results_10_percent.values()
all_model_results = all_model_results.transpose()
all_model_results

"""# compare the models"""

all_model_results['accuracy'] /= 100
all_model_results

all_model_results.plot(kind='bar', figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))

all_model_results.sort_values('f1', ascending=False)['f1'].plot(kind='bar', figsize=(10, 7))

all_model_results.sort_values('f1', ascending=False)['f1'][['11-nb_10_percent', '7-USE_10_percent_data']].plot(kind='bar', figsize=(10, 7))

"""# fine tuning tfhub USE on """

sentence_encoder_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=True,
                                        name='USE')

model_12 = tf.keras.models.Sequential([
  sentence_encoder_layer,
  layers.Dense(64, activation='relu'),
  layers.Dense(1, activation='sigmoid')   
], name='fine_tuned_USE')

model_12.compile(loss='binary_crossentropy',
                 optimizer=optimizers.Adam(),
                 metrics=['accuracy'])

model_12.summary()

model_12_feat_history = model_12.fit(train_sentences, 
                                     train_labels,
                                     epochs=5,
                                     validation_data=(val_sentences, val_labels),
                                     callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                            experiment_name='model_12_fine_tuned_USE')])

model_12_pred_probs = model_12.predict(val_sentences)
model_12_preds = tf.squeeze(tf.round(model_12_pred_probs))
model_12_pred_probs[:10], model_12_preds[:10]

model_12_results = calculate_results(y_true=val_labels,
                                     y_pred=model_12_preds)
all_model_results = all_model_results.transpose()
all_model_results['12-tfhub_USE_fine_tuned'] = model_12_results.values()

!tensorboard dev upload --logdir ./model_logs \
  --name 'nlp_models_comparison' \
  --description 'comparing models on kaggle introduction to nlp dataset' \
  --one_shot

"""* Done. View your TensorBoard at https://tensorboard.dev/experiment/M7YLVa1ESjq7yWaq4vVqDA/"""

all_model_results['12-tfhub_USE_fine_tuned']['accuracy'] /= 100

all_model_results.transpose().sort_values('f1', ascending=False).plot(kind='bar', figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))

"""# best model

* model_6: tfhub USE on all data
"""

test_df

train_df_shuffled

train_sentences, train_labels = train_df_shuffled['text'], train_df_shuffled['target']
test_sentences, test_id = test_df['text'], test_df['id']

model_6.compile(loss=losses.BinaryCrossentropy(),
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

best_model_history = model_6.fit(train_sentences,
                                 train_labels,
                                 epochs=5,
                                 callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                        experiment_name='best_model_tfhub_USE')])

best_model_pred_probs = model_6.predict(test_sentences)
best_model_preds = tf.squeeze(tf.round(best_model_pred_probs))
best_model_preds = tf.cast(best_model_preds, tf.int32)
best_model_pred_probs[:10], best_model_preds[:10]

submission = pd.DataFrame(
    {'id':test_id,
     'target':best_model_preds}
)
submission

## save to csv file
from google.colab import files

!mkdir 'results/'
submission.to_csv('results/USE_submission.csv', index=False)
files.download('results/USE_submission.csv')

len(test_id), len(test_sentences)

"""# ensemble"""

def give_predictions(model, sentences):
  pred_probs = model.predict(sentences)
  preds = tf.squeeze(tf.round(pred_probs))
  preds = tf.cast(preds, tf.int32)
  return preds

ensemble_preds = pd.DataFrame({
    'lstm':give_predictions(model_2, test_sentences),
    'conv1d':give_predictions(model_5, test_sentences),
    'USE':give_predictions(model_6, test_sentences)
})
ensemble_preds

ensemble_preds['ensemble'] = ensemble_preds.mode(axis=1)
ensemble_preds

"""# confusion_matrix of best_model on validation data"""

from helper_functions import make_confusion_matrix

make_confusion_matrix(y_true=val_labels,
                      y_pred=give_predictions(model_6, val_sentences),
                      classes=['disaster', 'not disaster'],
                      figsize=(10, 7),
                      text_size=20)
