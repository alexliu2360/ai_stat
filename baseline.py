
# coding: utf-8

# In[175]:


from keras import Sequential
from keras.callbacks import Callback
from keras.layers import Embedding, np
from keras.utils import to_categorical
from keras_applications.densenet import layers
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from keras import backend as K

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import jieba


# ### 导入数据

# In[19]:


data = pd.read_csv('../data/trainingset/sentiment_analysis_trainingset.csv')


# ### 数据探索

# In[191]:


list(data.columns)[2:]


# In[5]:


data['content'][0]


# In[6]:


list(jieba.cut(data['content'][0]))[1:-1]


# In[25]:


data.shape


# ### 数据分析

# #### 1、数据预处理

# In[15]:


# 准备停用词
with open('./stopwords.txt') as f:
    stoplist = f.read().split()
stoplist.append(str("\n"))
stoplist.append(" ")


# In[21]:


# data_head10
data_head10 = data.loc[:10, :]
data_head10['words'] = data_head10['content'].apply(lambda x: list(jieba.cut(x)))


# In[226]:


# 划分小额数据做训练以及验证
train_counts = 500
val_counts = 100
test_counts = 200
data_ls = data.loc[:train_counts + val_counts + test_counts-1, :]
data_ls['words'] = data_ls['content'].apply(lambda x: list(jieba.cut(x)))
y_cols = list(data_ls.columns)[2:]
print(data_ls.shape)


# In[230]:


def not_empty(s):
    return s and s.strip()

list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))


# In[ ]:


data_ls['others_willing_to_consume_again'][0] = 'p'
# print(data_ls['others_willing_to_consume_again'])
def in_list(x):
    if x in [-2, -1, -1, 0, 1, 2]:
        return x
    else:
        print(x)
list(filter(in_list, data_ls['others_willing_to_consume_again']))


# In[215]:


train = data_ls.loc[:train_counts-1, :]
val = data_ls.loc[train_counts:train_counts+val_counts-1, :]
test = data_ls.loc[train_counts+val_counts:train_counts+val_counts+test_counts-1, :]


# In[182]:


# 去掉停用词
def remove_stopword(data, stoplist):
    texts = []
    for index, row in data.iterrows():
        line = [word for word in list(row['words']) if word not in stoplist]
        texts.append(line)
    return texts

# 求每一个样本的最大长度
def get_texts_maxlen(texts):
    maxlen = 0
    for line in texts:
        if maxlen < len(line):
            maxlen = len(line)
    return maxlen


# In[183]:


texts = remove_stopword(data_ls, stoplist)
# 利用keras的Tokenizer进行onehot，并调整未等长数组
max_words = 50000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
data_w = tokenizer.texts_to_sequences(texts)
maxlen=get_texts_maxlen(texts)
data_T = sequence.pad_sequences(data_w, maxlen=maxlen)
data_T.shape


# In[184]:


# 数据划分，重新划分为训练集，测试集和验证集
dealed_train = data_T[:train_counts]
dealed_val = data_T[train_counts:(train_counts + val_counts)]
dealed_test = data_T[(train_counts + val_counts):]


# In[185]:


print('train:', dealed_train.shape)
print('val:', dealed_val.shape)
print('test:', dealed_test.shape)


# #### 2、建立模型

# In[186]:


def build_model():
    model = Sequential()
    embedding_dim = 128
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    #     model.add(Dropout(0.5))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    #     model.add(Dropout(0.5))
    model.add(layers.GlobalMaxPooling1D())
    #     model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model


# In[187]:


class Metrics(Callback):    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1)
        return 

    
def train_CV_CNN(train_x=dealed_train, test_x=dealed_test, val_x=dealed_val, y_cols=y_cols, debug=False, folds=2):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    F1_scores = 0
    F1_score = 0
    metrics = Metrics()
    if debug:
        y_cols = ['location_traffic_convenience']
    for index, col in enumerate(y_cols):
        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0
        result = {}
        for i in range(1):
            X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=100 * i)
            y_train_onehot = to_categorical(y_train)
            y_test_onehot = to_categorical(y_test)
            history = model.fit(X_train, y_train_onehot, epochs=20, batch_size=64, 
                                validation_data=(X_test, y_test_onehot),callbacks=[metrics])

            # 预测验证集和测试集
            y_val_pred = model.predict(val_x)
            y_test_pred += model.predict(test_x)

            y_val_pred = np.argmax(y_val_pred, axis=1)

            F1_score = f1_score(y_val_pred, val_y, average='macro')
            F1_scores += F1_score

            print(col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred - 2
    print('all F1_score:', F1_scores / len(y_cols))
    return result


# In[188]:


train_CV_CNN()
