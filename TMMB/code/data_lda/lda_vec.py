import json
import pickle
import jieba
import numpy as np
def segment(text):
    seg_list = jieba.cut(text, cut_all=False)
    seg_result = ' '.join(seg_list)
    return seg_result

with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)


file = 'your_file\\test.json'

with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Traverse the JSON object and add "lda": xxxx.
i = 0
for item in data:
    # i+=1
    doc = item['content']
    doc_bow = dictionary.doc2bow(segment(doc).split())
    doc_topics = lda_model.get_document_topics(doc_bow)

    pros = np.zeros(11)
    for topic, probability in doc_topics:
        pros[topic - 1] = float(probability)

    item['pro_vec'] = pros.tolist()

# Write the modified JSON data to a file
with open('your_file\\test.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4,  ensure_ascii=False)