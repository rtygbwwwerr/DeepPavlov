
import pandas as pd
import gensim
from gensim.models import word2vec
import multiprocessing

def train_w2v(corpus_path, model_path):
    sentences = word2vec.LineSentence(corpus_path)
    # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
    sg = 0
    
    model = gensim.models.Word2Vec(sentences, size=100, min_count=5, null_word=-2,
                                   sg=sg, workers=multiprocessing.cpu_count())
    model.save(model_path)

if __name__ == "__main__":
    
    train_w2v('../download/ch_anti_cls/anti.txt', '../download/embeddings/anti_char_CBOW_gensim.emb')
    
#     f = open('../download/chsentiment/test_with_label/test.label.cn.txt', encoding='utf-8')
#     texts = []
#     labels = []
#     for line in f.readlines():
#         line = line.strip()
#         if line.find('</r') > -1 or line == '':
#             continue
#         
#         if line.find('<') > -1:
#             labels.append(line[-3])
#         else:
#             texts.append(line)

#     f = open('../download/chsentiment/train/cn_sample_data/sample.positive.txt', encoding='utf-8')
#     texts = []
#     labels = []
#     for line in f.readlines():
#         line = line.strip()
#         if line.find('<') > -1 or line.startswith('<') or line == '':
#             continue
#         texts.append(line)
#         labels.append('1')
#             
#     f = open('../download/chsentiment/train/cn_sample_data/sample.negative.txt', encoding='utf-8')
#      
#     for line in f.readlines():
#         line = line.strip()
#         if line.find('<') > -1 or line.startswith('<') or line == '':
#             continue
#         texts.append(line)
#         labels.append('0')
#             
#     df = pd.DataFrame(columns=['text', 'labels'])
#     df['text'] = texts
#     df['labels'] = labels
#     
#     df.to_excel('../download/chsentiment/train.xlsx', index=False)