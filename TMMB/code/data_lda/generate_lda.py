import jieba
from gensim import corpora
import pickle

def segment(text):
    seg_list = jieba.cut(text, cut_all=False)
    seg_result = ' '.join(seg_list)
    return seg_result


with open('train_texts.pkl', 'rb') as f:
    texts = pickle.load(f)

# Build a corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# lda_model = models.LdaModel(corpus=corpus,
#                             id2word=dictionary,
#                             num_topics=11,
#                             # iterations=50,
#                             passes=20,
#                             # decay=0.5
#                             )
#
# with open('lda_model.pkl', 'wb') as f:
#     pickle.dump(lda_model, f)
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
print('finished!')