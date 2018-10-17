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
import re

import jieba
import os
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.preprocessing import sequence
import numpy as np
from app.utils import setting, load_train_set, load_valid_set, load_test_set
import pandas as pd
from app.utils.setting import WORD_DICT_CORPUS, TEXT_CUT_DIR
from app.utils.setting import STOP_WORDS_PATH


# from app.utils.setting import USER_WORDS_PATH


# TODO: 分词重点关注和处理
def word_cut(X):
    """ Cut Corpus with jieba

    Parameters
    ----------
    X: DataFrame or Series
      Input corpus

    Returns
    ----------
    x_cut: Series
      Output corpus cut by jieba
    """
    # 把停用词做成字典
    stop_words = {}
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f_stop:
        for each_word in f_stop:
            stop_words[each_word.strip()] = each_word.strip()
    print(stop_words)
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    x_cut = pd.Series([0] * X.shape[0])
    for ind in X.index:
        line = X[ind].strip()
        line_sub = re.sub('[\s+\.\/_,$%^*();；:-【】+\'\']+|[+——！，;:。、~#￥%&*（）]+', '', line)
        word_list = jieba.lcut(line_sub)
        cut_str = ''
        for word in word_list:
            if word not in stop_words:
                cut_str += word
                cut_str += ' '
        x_cut[ind] = cut_str.split(' ')
    return x_cut


def train_preprocess():
    train_text_cut_path = os.path.join(TEXT_CUT_DIR, 'sentiment_analysis_train_set_text_cut.csv')
    valid_text_cut_path = os.path.join(TEXT_CUT_DIR, 'sentiment_analysis_valid_set_text_cut.csv')
    train_set = load_train_set()
    valid_set = load_valid_set()
    if not os.path.exists(train_text_cut_path) and not os.path.exists(valid_text_cut_path):
        x_train_cut = word_cut(train_set.iloc[:, 1])
        x_valid_cut = word_cut(valid_set.iloc[:, 1])
        x_train_cut.to_csv(train_text_cut_path, encoding='utf-8')
        x_valid_cut.to_csv(valid_text_cut_path, encoding='utf-8')
    else:
        x_train_cut = pd.read_csv(open(train_text_cut_path, encoding='utf-8'), header=None, index_col=0).iloc[:, 0]
        x_valid_cut = pd.read_csv(open(valid_text_cut_path, encoding='utf-8'), header=None, index_col=0).iloc[:, 0]
    return x_train_cut, x_valid_cut, train_set, valid_set


def submit_preprocess():
    test_text_cut_path = os.path.join(TEXT_CUT_DIR, 'sentiment_analysis_test_set_text_cut.csv')
    test_set = load_test_set()
    if not os.path.exists(test_text_cut_path):
        x_test_cut = word_cut(test_set.iloc[:, 1])
        x_test_cut.to_csv(test_text_cut_path, encoding='utf-8')
    else:
        with open(test_text_cut_path, 'r', encoding='utf-8') as fp:
            x_train_cut = fp.readlines()
            print(x_train_cut)
        x_test_cut = pd.read_csv(open(test_text_cut_path, encoding='utf-8'), header=None)
    return x_test_cut, test_set


def word2vec_train(X):
    if os.path.exists(setting.WORD_VECTOR_CORPUS):
        model = Word2Vec.load(setting.WORD_VECTOR_CORPUS)
    else:
        model = Word2Vec(size=setting.VOCABULARY_VECTOR_DIM,
                         min_count=setting.MIN_WORD_COUNTS,
                         window=setting.WINDOW_SIZE,
                         workers=setting.CPU_COUNTS,
                         iter=setting.N_ITERATIONS)
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(setting.WORD_VECTOR_CORPUS)
    index_dict, word_vectors, x_combined = create_dictionaries(model=model, X=X)
    with open(WORD_DICT_CORPUS, 'w', encoding='utf-8') as f:
        for word, ind in index_dict.items():
            f.writelines('{}:{}\n'.format(word, ind))
    return index_dict, word_vectors, x_combined


def parse_dataset(X_words, w2v_ind):
    """ Words become integers
    """
    data = []
    for words in X_words:
        new_txt = []
        for word in words:
            try:
                new_txt.append(w2v_ind[word])
            except KeyError:
                new_txt.append(0)
        data.append(new_txt)
    return data


def create_dictionaries(model=None, X=None):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
       Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    """
    if (X is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2v_ind = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2v_ind.keys()}  # 所有频数超过10的词语的词向量
        X = parse_dataset(X, w2v_ind)
        # 每个句子所含词语对应的索引，所有句子中含有频数小于10的词语，索引为0
        X = sequence.pad_sequences(X, maxlen=setting.VOCABULARY_MAXLEN)
        return w2v_ind, w2vec, X
    else:
        print('No data provided...')


def get_model_data(index_dict, word_vectors):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, setting.VOCABULARY_VECTOR_DIM))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return n_symbols, embedding_weights


def predict_create_dictionaries(X=None):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
       Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    """
    if X is not None:
        word_dict = word2num_dict(WORD_DICT_CORPUS)
        x_transformed = parse_dataset(X, word_dict)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        x_transformed = sequence.pad_sequences(x_transformed, maxlen=setting.VOCABULARY_MAXLEN)
        return x_transformed
    else:
        print('No data provided...')


def input_transform(x_cut):
    x_transformed = predict_create_dictionaries(x_cut)
    print(x_transformed)
    return x_transformed


def id2num_dict(file_path):
    domain_dict = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        for pair_ in f.readlines():
            domain_dict[int(pair_.split(':')[0].strip())] = int(pair_.split(':')[1].strip())
    return domain_dict


def word2num_dict(file_path):
    domain_dict = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        for pair_ in f.readlines():
            domain_dict[pair_.split(':')[0].strip()] = int(pair_.split(':')[1].strip())
    return domain_dict


if __name__ == '__main__':
    submit_preprocess()
