from __future__ import print_function
import glob
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model



import pickle
with open('c:\NLP学习备用\corpus_list','wb') as f:
    pickle.dump(corpus,f)

with open(r'c:\NLP学习备用\corpus_clean_list','rb') as f:
    corpus_clean = pickle.load(f)


# 基本参数
maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/vocab.txt'

# 训练样本。THUCNews数据集，每个样本保存为一个txt。
txts = glob.glob('/root/thuctc/THUCNews/*/*.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, txt in self.sample(random):
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                token_ids, segment_ids = tokenizer.encode(
                    content, title, max_length=maxlen
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss





# model.compile(optimizer=Adam(1e-5))
#
# model.summary()

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)

if __name__ == '__main__':
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    #
    # model.load_weights('../content/drive/My Drive/best_model_title_two.weights')

    txt = "5月12日23时30分，中央纪委国家监委官网发布重磅，原中国船舶重工集团有限公司党组书记、董事长胡问鸣涉嫌严重违纪违法，目前正接受中央纪委国家监委纪律审查和监察调查。"

    autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

    print(autotitle.generate(txt))
