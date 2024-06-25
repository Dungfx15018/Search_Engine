import os
import math
import json
from collections import defaultdict
import string
from datasets import load_dataset
class TFIDF_Search:
    def __init__(self):
        self.dataset = load_dataset('ILT37/viwiki', split='train')

    def clean_word(self,w):
        letters = set('aáàảãạăaáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz0123456789')
        new_w = ''
        for letter in w:
            if letter.lower() in letters or letter == '.':
                new_w += letter.lower()
        return new_w

    def preprocessing(self, docs):
        new_docs = []
        new_doc = ' '
        for i in range(len(docs)):
            doc = docs[i]
            doc = doc.replace('\n', ' ').replace('==', ' ')
            words = doc.split()
            for j in range(len(words)):
                word = self.clean_word(words[j])
                words[j] = word
            new_doc = new_doc.join(words)
            new_docs.append(new_doc)
        return new_docs
    def load_data(self,num_docs):
        self.docs = self.preprocessing(self.dataset['text'][:num_docs])
        return self.docs

    def inverse_tf_idf(self,docs):

        stats = {
            "words": {},
            "docs" : {}
        }
        for i, doc in enumerate(docs):
            if i not in stats['docs']:
                stats['docs'][i] = defaultdict(int)

            for word in doc.split(' '):
                if word not in stats['words']:
                    stats['words'][word] = {i}
                else:
                    stats['words'][word].add(i)

                stats['docs'][i][word] += 1
        return stats

    def rounding(self,num):
        return math.floor(num*1000)/1000

    def get_tf(self, num):
        return self.rounding(math.log10(num+1))

    def compute_tf_idf(self,docs):

        words = self.inverse_tf_idf(docs)['words'].keys()

        # Calculation IDF
        idf = defaultdict(float)
        N = len(docs)

        for word in words:
            df = len(self.inverse_tf_idf(docs)['words'][word])

            idf[word] = math.log10(N/df)

        tf_idf_list = defaultdict(lambda: defaultdict(float))

        ds = defaultdict(float)

        for doc in self.inverse_tf_idf(docs)['docs']:
            d = 0
            for word in words:

                tf = self.get_tf(self.inverse_tf_idf(docs)['docs'][doc][word])

                if tf!=0:

                    tf_idf = tf * idf[word]

                    d+= tf_idf**2

                    tf_idf_list[word][doc] = tf_idf

            d_denominator = math.sqrt(d)
            ds[doc] = self.rounding(d_denominator)
        return tf_idf_list, ds
    def save_data(self,tf_idf_list,ds):
        with open('tf_idf_list1.json', 'w') as outfile:
            json.dump(tf_idf_list, outfile)
        print('Saved tf_idf_list1!!!!')

        with open('ds1.json', 'w') as outfile:
            json.dump(ds, outfile)
        print('Saved ds1')

    def get_file(self,num_docs):
        with open('docs1.json', 'w') as outfile:
            documents = {
                "docs": self.load_data(num_docs)
            }
            json.dump(documents, outfile)
        print('Saved docs1')
        tf_idf_list, ds = self.compute_tf_idf((self.load_data(num_docs)))
        self.save_data(tf_idf_list, ds)

        return tf_idf_list, ds
tfidf = TFIDF_Search()
#print(tfidf.load_data(num_docs=1500))
#print(tfidf.inverse_tf_idf(tfidf.load_data(num_docs=1500)))
#print(tfidf.compute_tf_idf(tfidf.load_data(num_docs=10)))
tf_idf_list, ds = tfidf.get_file(1500)
