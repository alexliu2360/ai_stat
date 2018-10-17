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
from configparser import ConfigParser

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROJECT_NAME = os.path.basename(PROJECT_PATH)
CONFIG_PATH = os.path.join(PROJECT_PATH, 'cfg/config.ini')
cf = ConfigParser()
cf.read(CONFIG_PATH)
VERSION = cf.get('ReleaseNotes', 'VERSION')

MODELS_SAVE_DIR = os.path.join(u'{}'.format(PROJECT_PATH), 'result/models_{}'.format(VERSION))
TEXT_CUT_DIR = os.path.join(u'{}'.format(PROJECT_PATH), 'result/text_cut')
F1_SCORE_PATH = os.path.join(u'{}'.format(PROJECT_PATH), 'result/f1_score_{}.txt'.format(VERSION))
SUBMIT_RESULT_PATH = os.path.join(u'{}'.format(PROJECT_PATH), 'result/submission_{}.csv'.format(VERSION))
STOP_WORDS_PATH = os.path.join(u'{}'.format(PROJECT_PATH), 'cfg/stop_words.txt')
USER_WORDS_PATH = os.path.join(u'{}'.format(PROJECT_PATH), 'cfg/user_dict.txt')
if not os.path.exists(MODELS_SAVE_DIR):
    os.makedirs(MODELS_SAVE_DIR)
if not os.path.exists(TEXT_CUT_DIR):
    os.makedirs(TEXT_CUT_DIR)
VOCABULARY_SIZE = 2000
VOCABULARY_MAXLEN = 500
WORD_VECTOR_CORPUS = os.path.join(MODELS_SAVE_DIR, 'text_word2vec_model_{}.pkl'.format(VERSION))
WORD_DICT_CORPUS = os.path.join(MODELS_SAVE_DIR, 'text_word2vec_dict_{}.csv'.format(VERSION))
VOCABULARY_VECTOR_DIM = 500
BATCH_SIZE = 1000
N_EPOCH = 1
MIN_WORD_COUNTS = 10
WINDOW_SIZE = 5
CPU_COUNTS = 4
N_ITERATIONS = 2


ALGORITHM = 'LSTM'  # LSTM, CNN, DNN
