# coding=utf-8
# ------------------------------------------------------------
#  版本：0.1
#  版权：****
#  模块：****
#  功能：****
#  语言：Python3.6
#  作者：****<****@aconbot.com.cn>
#  日期：2018-07-18
# ------------------------------------------------------------
#  修改人：****<****@aconbot.com.cn>
#  修改日期：2018-07-18
#  修改内容：创建
# ------------------------------------------------------------
import os

from keras.engine.saving import load_model
from sklearn.metrics import f1_score
from app.models.algorithm.text_cnn import ChatTextCNN
from app.models.algorithm.text_dnn import ChatDNN
from app.models.algorithm.text_lstm import ChatTextLSTM
from app.models.clean import input_transform, word2vec_train, get_model_data
from app.models.clean.text_clean import train_preprocess, submit_preprocess
import pandas as pd
import numpy as np
from app.utils.setting import MODELS_SAVE_DIR, VERSION, VOCABULARY_VECTOR_DIM, ALGORITHM, F1_SCORE_PATH, \
    SUBMIT_RESULT_PATH


def model_select(n_symbols, embedding_weights):
    if ALGORITHM is 'CNN':
        return ChatTextCNN(input_dim=n_symbols, embedding_dim=VOCABULARY_VECTOR_DIM,
                           embedding_weights=embedding_weights)
    elif ALGORITHM is 'LSTM':
        return ChatTextLSTM(input_dim=n_symbols, embedding_dim=VOCABULARY_VECTOR_DIM,
                            embedding_weights=embedding_weights)
    elif ALGORITHM is 'DNN':
        return ChatDNN(input_dim=n_symbols, embedding_dim=VOCABULARY_VECTOR_DIM,
                       embedding_weights=embedding_weights)
    else:
        return ChatTextLSTM(input_dim=n_symbols, embedding_dim=VOCABULARY_VECTOR_DIM,
                            embedding_weights=embedding_weights)


def sentiment_train_manager():
    print('Train Text Preprocess')
    x_train_cut, x_valid_cut, train_set, valid_set = train_preprocess()
    print('Train Text Word Embedding')
    index_dict, word_vectors, x_combined = word2vec_train(x_train_cut)
    n_symbols, embedding_weights = get_model_data(index_dict, word_vectors)
    x_valid = input_transform(x_valid_cut)
    x_train = x_combined
    print('Text Words Length:{}'.format(len(index_dict)))
    print('Model Select:{}'.format(ALGORITHM))
    text_train_model = model_select(n_symbols, embedding_weights)
    f1_score_dict = dict()
    print('Start Model Training....')
    col_len = len(train_set.columns[2:])
    for i, col in enumerate(train_set.columns[2:]):
        print('{} column is training， finish {}%!'.format(col, float(i) / float(col_len) * 100))
        y_train = train_set[col] + 2
        y_valid = valid_set[col] + 2
        text_train_model.train(x_train, y_train, x_valid, y_valid)
        y_valid_pred = pd.Series([0] * x_valid.shape[0])
        for ind in range(x_valid.shape[0]):
            y_pred = np.argmax(text_train_model.model.predict(x_valid[ind].reshape(1, -1)))
            y_valid_pred[ind] = y_pred
        f1_score_dict[col] = f1_score(y_valid, y_valid_pred, average='macro')
        print('{} F1 Score:{}'.format(col, f1_score_dict[col]))
        model_save_path = os.path.join(MODELS_SAVE_DIR, '{}_model_{}_{}.h5'.format(ALGORITHM.lower(), col, VERSION))
        text_train_model.model.save(model_save_path)
    f1_score_mn = np.mean(list(f1_score_dict.values()))
    with open(F1_SCORE_PATH, 'w', encoding='utf-8') as fp:
        for col, f1_score_ in f1_score_dict.items():
            fp.writelines('{}:{}\n'.format(col, f1_score_))
    print('Train Finished, F1 Score:{}'.format(f1_score_mn))
    return True


def load_predict_model(columns):
    model_dict = dict()
    for col in columns:
        model_path = os.path.join(MODELS_SAVE_DIR, '{}_model_{}_{}.h5'.format(ALGORITHM.lower(), col, VERSION))
        model_dict[col] = load_model(model_path)
    return model_dict


def sentiment_submit_manager():
    x_submit_cut, x_submit_set = submit_preprocess()
    x_submit = input_transform(x_submit_cut)
    predict_model_dict = load_predict_model(x_submit_set.columns)
    for col in x_submit_set.columns[2:]:
        print('{} columns is predicting'.format(col))
        y_submit_pred = pd.Series([0] * x_submit.shape[0])
        for ind in range(x_submit.shape[0]):
            y_pred = np.argmax(predict_model_dict[col].predict(x_submit[ind].reshape(1, -1)))
            y_submit_pred[ind] = y_pred - 2
        x_submit_set[col] = y_submit_pred
    x_submit_set.to_csv(SUBMIT_RESULT_PATH, encoding='utf-8')
    print(x_submit)
    return True


if __name__ == '__main__':
    sentiment_train_manager()
    sentiment_submit_manager()
