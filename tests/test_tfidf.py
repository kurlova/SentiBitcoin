import unittest
from data_processor import *


class TestTFIDF(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            # positive
            ['bitcoin', 'good', 'christmas', 'rest', '–', 'rise', '10', '%', 'http', ':', '//t.co/acwwahxilc', 'http',
             ':', '//t.co/zi9755st1e'],
            ['bitcoin', 'big', 'larp', "'s", "'s", 'cool', '.'],
            ['gold', 'great', 'way', 'preserve', 'wealth', ',', 'hard', 'move', 'around', '.'],
            ['love', '#', 'bitcoin', 'community', '!'],
            ['good', 'luck', 'bitcoin', '.'],
            ['great', 'step', 'forward', '#', 'crypto', '#', 'bitcoin', 'well', 'electric', 'car', 'owner', '.'],
            ['bitcoin', 'f', 'good'],
            ['bitcoin', 'good'],

            # negative
            ['yea', 'suck', '.'],
            ['hold', '...', '.', 'useless', '.'],
            ['bitcoin', 'ridiculous', '.'],
            ['bitcoin', 'bad'],
            ['bitcoin', 'very', 'bad'],
            ['suck', 'very', 'much'],
            ['hate', 'bitcoin'],
            ['only', 'have', 'hate']
        ]
    def test_tf(self):
        tweet = ['complete', 'financial', 'autonomy', 'independence', 'for']
        tweet_counter = collections.Counter(tweet)
        res = compute_tf(tweet_counter)
        # print(res)

    def test_idf(self):
        corpus = [
            ['complete', 'financial', 'autonomy', 'independence', 'for'],
            ['every', 'bug', 'found', 'make', 'developer', 'smarter'],
            ['financial', 'autonomy']
        ]
        words = ['financial', 'found']

        for word in words:
            res = compute_idf(word, corpus)
            # print(word, res)

    def test_tfidf(self):
        documents = []

        for tweet in self.corpus:
            tfidf_dict = dict()
            tweet_counter = collections.Counter(tweet)
            tf_data = compute_tf(tweet_counter)

            for word in tf_data:
                tfidf_dict[word] = tf_data[word] * compute_idf(word, self.corpus)

            documents.append(tfidf_dict)

        all_words = []
        for doc in documents:
            for key, value in doc.items():
                all_words.append((key, value))

        all_words.sort(key=lambda el: el[1])

        for el in all_words:
            print(el)