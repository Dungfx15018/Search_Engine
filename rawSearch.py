import json
class TFIDF():
    def __init__(self):
        # Load data
        with open('docs.json', 'r', encoding='utf-8') as f:
            self.docs = json.load(f)

        with open('ds.json', 'r', encoding='utf-8') as f:
            self.ds = json.load(f)
        with open('tf_idf_list.json', 'r', encoding='utf-8') as f:
            self.tf_idf_list = json.load(f)

    def search(self, q, k):
        results = []

        finals = []

        for doc, document in self.docs.items():

          docss = document

          for i in range(len(docss)):

            score = 0

            for t in q.split():

              t = t.lower()

              score += self.tf_idf_list[t][str(i)] / self.ds[str(i)]
            finals.append((score, i))

            finals.sort(key=lambda x: -x[0], reverse = False)

            finals = finals[:k]

            indices = [x for x in finals]

          for i in indices:

            results.append(docss[i[1]])

        results = [{"text": r} for r in results]

        return results
