import jieba
from gensim import corpora, models
import pickle
import json
import matplotlib.pyplot as plt
import math
import numpy as np
from gensim.models import CoherenceModel


def main():
    with open('train_texts.pkl', 'rb') as f:
        texts = pickle.load(f)

    # 建立语料库
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    per_list = []
    x_values = []
    co_list = []

    def coherence(num_topics):
        lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=num_topics,
                                    # iterations=50,
                                    passes=5,
                                    # decay=0.5
                                    )
        ldacm = CoherenceModel(model=lda_model,
                               texts=texts,
                               dictionary=dictionary,
                               coherence='c_v')
        return ldacm.get_coherence()

    def perplexity(num_topics):
        lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=num_topics,
                                    # iterations=50,
                                    passes=5,
                                    # decay=0.5
                                    )
        perplexity = np.exp2(-(lda_model.log_perplexity(corpus)))
        return perplexity

    for i in range(1, 31):
        co_list.append(coherence(i))
        #per_list.append(perplexity(i))
        print('第{}项已完成'.format(i))
        x_values.append(i)

    plt.title('topic-coherence')
    plt.xlabel('topic')
    plt.ylabel('coherence')

    y_values = co_list
    plt.plot(x_values, y_values, 'o-', markersize=5)
    plt.show()


if __name__ == '__main__':
    main()