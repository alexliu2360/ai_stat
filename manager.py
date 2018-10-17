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
from app.models import sentiment_train_manager, sentiment_submit_manager


def manager():
    sentiment_train_manager()
    sentiment_submit_manager()
    return True


if __name__ == '__main__':
    manager()

