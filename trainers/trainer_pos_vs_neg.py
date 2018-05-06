# {retweet_count: {$eq: 0},'entities.urls': {$size: 0},truncated: {$eq: false}}     {_id: -1}

import nltk
import traceback

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC  # Support Vector Machines
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from db_manager import Connection
from data_processor import *
from utils import serialize, PCKL_DATADIR, PCKL_POSNEG_FILENAME_START


if __name__ == "__main__":
    # set up connection
    conn = Connection()
    collection = conn.get_english_collection()

    try:
        # WORK WITH TRAINING DATA
        # work with positive tweets
        positive = TrainTextPreprocessor(filename="..\\training_data\\positive.txt", db_collection=collection,
                                        category='POS')
        positive.process_data()
        positive_features = positive.get_most_frequent(100)
        print('positive features\t', positive_features)
        positive_tweets_tokens = positive.category_tokens
        print('positive tweets` tokens\t', positive_tweets_tokens)
        print(f'len of positive tokens: {len(positive_tweets_tokens)}')
        positive_tweets = positive.tweets_lemmas
        print('positive tweets\t', positive_tweets)

        # work with negative tweets
        negative = TrainTextPreprocessor(filename="..\\training_data\\negative.txt", db_collection=collection,
                                        category='NEG')
        negative.process_data()
        negative_tweets_tokens = negative.category_tokens
        negative_tweets = negative.tweets_lemmas

        # word features is combined and shuffled tokens from both negative and positive tweets
        word_features = combine_and_shuffle(positive_tweets_tokens, negative_tweets_tokens)[:3000]
        print('word features\t', word_features)
        print(f'length of word features: {len(word_features)}')

        # documents are combined and shuffled positive and negative tweets (grouped tokens)
        documents = combine_and_shuffle(positive_tweets, negative_tweets)
        print('documents[0]\t', documents[0])

        featuresets = []
        for tweet, category in documents:
            featuresets.append((find_features(tweet, word_features), category))
        print('featuresets length', len(featuresets))

        training_set = featuresets[:600]
        testing_set = featuresets[600:]

        print('training_set[0]\t', training_set[0])
        print('testing_set[0]\t', testing_set[0])

        # NAIVE BAYES CLASSIFIER
        NBC_classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Original NB classifier accuracy percent:", (nltk.classify.accuracy(NBC_classifier, testing_set)) * 100)

        NBC_classifier.show_most_informative_features(15)

        # # MULTINOMIAL NAIVE BAYES CLASSIFIER
        # MNB_classifier = SklearnClassifier(MultinomialNB())  # play with alpha parameter: alpha=0.7
        # MNB_classifier.train(training_set)
        # print("Multinomial NB classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,
        #                                                                              testing_set)) * 100)
        #
        # # BERNOULLI NAIVE BAYES CLASSIFIER
        # BNB_classifier = SklearnClassifier(BernoulliNB())
        # BNB_classifier.train(training_set)
        # print("Bernoulli Naive Bayes classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier,
        #                                                                                     testing_set)) * 100)
        #
        # # LOGISTIC REGRESSION CLASSIFIER
        # LogReg_classifier = SklearnClassifier(LogisticRegression())
        # LogReg_classifier.train(training_set)
        # print("Logistic Regression classifier accuracy percent:", (nltk.classify.accuracy(LogReg_classifier,
        #                                                                                   testing_set)) * 100)
        #
        # # SGDIClassifier
        # SGDI_classifier = SklearnClassifier(SGDClassifier())
        # SGDI_classifier.train(testing_set)
        # print("SGDI classifier accuracy percent:", (nltk.classify.accuracy(SGDI_classifier,
        #                                                                    testing_set)) * 100)
        #
        # # SVC (Support Vector Machine) CLASSIFIER
        # SVC_classifier = SklearnClassifier(SVC())
        # SVC_classifier.train(testing_set)
        # print("SVC classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)
        #
        # LINEAR SVC (Support Vector Machine) classifier
        # LinearSVC_classifier = SklearnClassifier(LinearSVC())
        # LinearSVC_classifier.train(testing_set)
        # print("Linear SVC classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,
        #                                                                          testing_set)) * 100)

        #
        # # NUSVC CLASSIFIER
        # NuSVC_classifier = SklearnClassifier(NuSVC())
        # NuSVC_classifier.train(testing_set)
        # print("NuSVC classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,
        #                                                                     testing_set)) * 100)

        # pickle things
        serialize(documents, PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "documents"))
        serialize(word_features, PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "word_features"))
        serialize(featuresets, PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "featuresets"))
        serialize(NBC_classifier, PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "NBC_classifier"))
    except:
        traceback.print_exc()

    # close connection
    conn.close_connection()
