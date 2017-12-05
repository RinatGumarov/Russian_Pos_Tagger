import string
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

try:
    from lxml import etree
except ImportError:
    print('lxml not found. xml.etree.ElementTree will be used')
    import xml.etree.ElementTree as etree


class DataParser:
    def __init__(self, limit):
        self._limit = limit

    def read_data(self):
        tagget_sentences = []
        if self._limit > 4062:
            self._limit = 4062
        for i in range(self._limit):
            if (i == 33):  # missing 34.xml
                continue
            filename = 'annot/{}.xml'.format(i + 1)
            tagget_sentences.extend(self.read_xml('annot/{}.xml'.format(i + 1)))
        return tagget_sentences

    def read_xml(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        paragraphs = tree.find('paragraphs')
        tagget_sentences = []
        for paragraph in paragraphs:
            for sentence in paragraph:
                tokens = sentence.find('tokens')
                sent = []
                for token in tokens:
                    word = token.attrib['text']
                    try:
                        pos = token.find('tfr').find('v').find('l').find('g').attrib['v']
                        sent.append((word, pos))
                    except:
                        continue
                tagget_sentences.append(sent)
        return tagget_sentences


class Tagger:
    def __init__(self, parser):
        self._parser = parser
        self._tt_data_limit = .75
        self._tagget_sentences = parser.read_data()
        self._clf = self._train_model()

    def _features(self, sentence, index):
        features = {}
        word = sentence[index]
        prevw = '' if index == 0 else sentence[index - 1]
        nextw = '' if index == len(sentence) - 1 else sentence[index + 1]
        return {
            'word': word,
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': word[0].upper() == word[0],
            #         'is_all_caps': word.upper() == word,
            #         'is_all_lower': word.lower() == word,
            'prefix-1': word[0],
            'prefix-2': word[:2],  # some specific signs
            'suffix-1': word[-1],  # y
            'suffix-2': word[-2:],  # ed, ly
            'suffix-3': word[-3:],  # ing
            'suffix-4': word[-4:],
            #         'digit_start': word[0] in '0123456789',
            'vowel_last': word[-1] in 'aeiouy',
            'vowels': ''.join([i for i in word if i in 'аоиеёэыуюя']),
            'is_punctuation': word in string.punctuation,
            #         'prev_word': prevw,
            #         'next_word': nextw,
            'latin': word[0] in 'abcdefghijklmnopqrstuvwxyz',
            'prev-suffix-1': prevw[-1:],
            'prev-suffix-2': prevw[-2:],
            'prev-suffix-3': prevw[-3:],
            'next-suffix-1': nextw[-1:],
            'next-suffix-2': nextw[-2:],
            'next-suffix-3': nextw[-3:],
            #         'len_gt_3': len(word) > 3,
            #         'is_numeric': word.isdigit(),
            'contains_hypen': '-' in word,
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }

    def _remove_tags(self, ts):
        return [w for (w, t) in ts]

    def _transform_to_dataset(self, ts):
        X, y = [], []
        for tagged in ts:
            for index in range(len(tagged)):
                X.append(self._features(self._remove_tags(tagged), index))
                y.append(tagged[index][1])
        return X, y

    def _train_model(self):
        limit = (int)(len(self._tagget_sentences) * self._tt_data_limit)
        X, y = self._transform_to_dataset(self._tagget_sentences[:limit])
        #         X_test, y_test = self._transform_to_dataset(self._tagget_sentences[limit:])
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])
        clf.fit(X, y)
        return clf

    def get_tags(self):
        tags = []
        tt = []
        sentence = []
        for word in self._tagget_sentences:
            for (x, y) in word:
                if y not in tags:
                    tags.append(y)
                    tt.append((x, y))
        return tt

    def get_report(self):
        limit = (int)(len(self._tagget_sentences) * self._tt_data_limit)
        X, y = self._transform_to_dataset(self._tagget_sentences[limit:])
        return classification_report(y, self._clf.predict(X))

    def pos_tag(self, sentences):
        X = []
        tagget = []
        for s in sentences:
            for index in range(len(s)):
                X.append(self._features(s, index))
            y = self._clf.predict(X)
            paired = []
            for i in range(len(s)):
                paired.append((s[i], y[i]))
            tagget.append(paired)
            X = []
        return tagget

    def print_mistakes(self):
        limit = (int)(len(self._tagget_sentences) * self._tt_data_limit)
        X, y = self._transform_to_dataset(self._tagget_sentences[limit:])
        y_pred = self._clf.predict(X)
        for m, n, k in zip([x['word'] for x in X], y_pred, y):
            if (n != k):
                print(m, 'f:' + n, 't:' + k)


def get_tagger(limit=20):
    tagger = None
    try:
        with open('tagger.model.pkl', 'rb') as input:
            tagger = pickle.load(input)
    except:
        with open('tagger.model.pkl', 'wb') as output:
            tagger = Tagger(DataParser(limit))
            pickle.dump(tagger, output, pickle.HIGHEST_PROTOCOL)
    return tagger
