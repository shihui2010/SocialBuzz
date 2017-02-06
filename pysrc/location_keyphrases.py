"""
key-phrase/word representation for each location
"""


import pprint
import json
import copy
import os
import re
# from phrases import NGramPhrase
from text_utils import clean_data, UNK



class TextRank(object):
    """
    ranking keywords/keyphrases by tokenized docs
    """
    def __init__(self, docs):
        """
        docs should be cleaned and tokenized text
        """
        self.__matrix = dict()
        self.__scores = None
        for doc in docs:
            last = UNK
            for token in doc:
                if token != UNK and token != '' and last != UNK:
                    tmp = self.__matrix.setdefault(last, dict())
                    tmp[token] = tmp.setdefault(token, 0) + 1
                    tmp = self.__matrix.setdefault(token, dict())
                    tmp[last] = tmp.setdefault(last, 0) + 1
                last = token

    def __unify(self):
        for key in self.__matrix:
            total = float(sum(self.__matrix[key][key2] for key2 in self.__matrix[key]))
            if total == 0:
                continue
            self.__matrix[key] = {key2: self.__matrix[key][key2] / total for key2 in self.__matrix[key]}
    
    def scoring(self, damping=0.85, max_iter=30, converage=0.00):
        self.__unify()
        self.__scores = {key: 1.0/len(self.__matrix) for key in self.__matrix}
        n_iter = 0
        while n_iter < max_iter:
            last_score = copy.copy(self.__scores)
            should_stop = True
            for key in last_score:
                self.__scores[key] = sum(last_score[t] * self.__matrix[t][key] for t in self.__matrix[key]) * damping + (1 - damping) / len(self.__matrix)
                # print key, sum(last_score[t] * self.__matrix[key][t] for t in self.__matrix[key])
                if should_stop and (self.__scores[key] - last_score[key]) / last_score[key] > converage:
                    should_stop = False
            if should_stop:
                break
            n_iter += 1
        return self.__scores



class SegSentence(object):
    """
    mining phrases in docs based on frequency
    re-segment tokenized texts based on phrases instead of words
    """
    def __init__(self, docs):
        self.__unigram, self.__bigram = dict(), dict()
        self.__min_sup = 20 if len(docs) > 100 else 0.4 * len(docs)
        self.__threshold = 1.4
        self.__docs = []
        self.__segments = []
        for doc in docs:
            self.__docs.append("|".join(doc))
            last = UNK
            for token in doc:
                if token != UNK:
                    self.__unigram[token] = self.__unigram.setdefault(token, 0) + 1
                if token != UNK and last != UNK:
                    tmp = self.__bigram.setdefault(last, dict())
                    tmp[token] = tmp.setdefault(token, 0) + 1
                    tmp = self.__bigram.setdefault(token, dict())
                    tmp[last] = tmp.setdefault(last, 0) + 1
                last = token
        self.__total = sum(self.__unigram.values())
        self.__phrases = {2: self.__init_phrase()}
        self.find_phrase()
        phrases_set = set()
        for length in self.__phrases:
            for prefix in self.__phrases[length]:
                for postfix in self.__phrases[length][prefix]:
                    phrases_set.add(prefix + "|" + postfix)
        for doc in docs:
            seg, token, index = [], "", 0
            # print doc
            while index < len(doc):
                if doc[index] == UNK or doc[index] not in self.__phrases[2]:
                    # single word
                    if doc[index] != UNK:
                        seg.append(doc[index])
                    index += 1
                    continue
                # possible phrase
                token = doc[index]
                for length in range(1, 7):
                    if index + length >= len(doc) or token + "|" + doc[index + length] not in phrases_set:
                        # prefix + word not a phrase, stop expending
                        seg.append(token)
                        index += length
                        token = ""
                    else:
                        token += "|" + doc[index + length]
            self.__segments.append(seg)


    def __init_phrase(self):
        phrases = {}
        for key in self.__bigram:
            if self.__unigram[key] < self.__min_sup:
                continue
            for key2 in self.__bigram[key]:
                if self.__unigram[key2] < self.__min_sup or self.__bigram[key][key2] < self.__min_sup:
                    continue
                ratio = float(self.__bigram[key][key2] * self.__total
                              ) / (self.__unigram[key] * self.__unigram[key2])
                if ratio > self.__threshold:
                    phrases.setdefault(key, {})[key2] = float(self.__bigram[key][key2]) / self.__total
        return phrases


    def find_phrase(self):
        for length in range(2, 6):
            new_phrases = {}
            for key in self.__phrases[length]:
                for key2 in self.__phrases[length][key]:
                    if key2 not in self.__phrases[length]:
                        continue
                    prefix = key + "|" + key2
                    for key3 in self.__phrases[length][key2]:
                        phrase = prefix + "|" + key3
                        count = self.__count_phrase(phrase)
                        if count < self.__min_sup:
                            continue
                        ratio = float(count) / (self.__phrases[length][key][key2] * self.__unigram[key3])
                        if ratio > self.__threshold:
                            new_phrases.setdefault(prefix, {})[key3] = float(count) / self.__total
            if len(new_phrases) < 1:
                break
            self.__phrases[length + 1] = new_phrases

    def __count_phrase(self, phrase):
        count = 0
        for doc in self.__docs:
            count += len(re.findall(phrase, doc))
        return count

    def segments(self):
        return self.__segments



if __name__ == '__main__':
    with open(os.path.join(os.pardir, 'data', 'chicago75000s_min_assigned.json')) as fin:
        locations = []
        for line in fin:
            locations.append(json.loads(line))
    model = []
    for loc in locations:
        docs = []
        for t in loc["text"]:
            docs.append(clean_data(t))
        seger = SegSentence(docs)
        segs = seger.segments()
        ranks = TextRank(segs)
        keywords = sorted(ranks.scoring().items(), key=lambda x: -x[1])[:10]
        name = loc["name"] if "name" in loc else "unknown place"
        feature = {"center": [loc["lon"], loc["lat"]], "name": name, "num_doc": len(loc["text"]), 
                    "topwords": [item[0] for item in keywords]}
        model.append(feature)
    with open(os.path.join(os.pardir, "data", "model.json"), 'w') as fout:
        json.dump(model, fout) 

        

