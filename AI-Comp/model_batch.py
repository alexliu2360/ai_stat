from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import random

random.seed = 42
import pandas as pd
from tensorflow import set_random_seed

set_random_seed(42)
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.layers import *
from classifier_capsule import TextClassifier
from gensim.models.keyedvectors import KeyedVectors
import pickle
import gc


def getClassification(arr):
    arr = list(arr)
    if arr.index(max(arr)) == 0:
        return -2
    elif arr.index(max(arr)) == 1:
        return -1
    elif arr.index(max(arr)) == 2:
        return 0
    else:
        return 1


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = list(map(getClassification, self.model.predict(self.validation_data[0])))
        val_targ = list(map(getClassification, self.validation_data[1]))
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(_val_f1, _val_precision, _val_recall)
        print("max f1")
        print(max(self.val_f1s))
        return


data = pd.read_csv("preprocess/train_char.csv")
data["content"] = data.apply(lambda x: eval(x[1]), axis=1)

validation = pd.read_csv("preprocess/validation_char.csv")
validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)

model_dir = "model_capsule_char/"
maxlen = 1000
max_features = 20000
batch_size = 128
epochs = 1
tokenizer = text.Tokenizer(num_words=None)
tokenizer.fit_on_texts(data["content"].values)
with open('tokenizer_char.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
w2_model = KeyedVectors.load_word2vec_format("word2vec/chars.vector", binary=True, encoding='utf8',
                                             unicode_errors='ignore')
embeddings_index = {}
embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
word2idx = {"_PAD": 0}
vocab_list = [(k, w2_model.wv[k]) for k, v in w2_model.wv.vocab.items()]

for word, i in word_index.items():
    if word in w2_model:
        embedding_vector = w2_model[word]
    else:
        embedding_vector = None
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

column_list = ["",
               "location_traffic_convenience", "location_distance_from_business_district",
               "location_easy_to_find", "service_wait_time",
               "service_waiters_attitude", "service_parking_convenience",
               "service_serving_speed", "price_level",
               "price_cost_effective", "price_discount",
               "environment_decoration", "environment_noise",
               "environment_space", "environment_cleaness",
               "dish_portion", "dish_taste",
               "dish_look", "dish_recommendation",
               "others_overall_experience", "others_willing_to_consume_again"
               ]

column_batch_map = {1: ['ltc', 'ldfbd'],
                    2: ['letf', 'swt'],
                    3: ['swa', 'spc'],
                    4: ['ssp', 'pl'],
                    5: ['pce', 'pd'],
                    6: ['ed', 'en'],
                    7: ['es', 'ec'],
                    8: ['dp', 'dt'],
                    9: ['dl', 'dr'],
                    10: ['ooe', 'owta']}

X_train = data["content"][:1000].values
X_validation = validation["content"][:1000].values

list_tokenized_train = tokenizer.texts_to_sequences(X_train)
input_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

list_tokenized_validation = tokenizer.texts_to_sequences(X_validation)
input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)

i = 1

'''
      index
    0       1
i    
1   1(1)    2(2)
2   2(3)    3(4)
3   3(5)    4(6)
4   4(7)    5(8)
5
'''

for index, item in enumerate(column_batch_map[i]):
    real_index = 2 * i + index - 1

    y_train = pd.get_dummies(data[column_list[real_index]])[[-2, -1, 0, 1]].values
    y_val = pd.get_dummies(validation[column_list[real_index]])[[-2, -1, 0, 1]].values

    print("model" + str(real_index))
    model = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
    file_path = model_dir + "model_" + str(item) + "_{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
    metrics = Metrics()
    callbacks_list = [checkpoint, metrics]
    history = model.fit(input_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(input_validation, y_val), callbacks=callbacks_list, verbose=2)
    del model
    del history
    gc.collect()
    K.clear_session()
