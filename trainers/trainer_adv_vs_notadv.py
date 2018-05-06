from nltk import SklearnClassifier
from sklearn.svm import LinearSVC

from db_manager import Connection
from data_processor import *
from utils import serialize


def get_training_ids_ms(*args):
    """
    Get IDs of training data in database from multiple sources

    :return: list with tweet IDs
    """
    result = []

    for filename in args:
        with open(f"{filename}", "r") as f:
            ids = f.readlines()
            ids = [tweet_id.rstrip('\n') for tweet_id in ids]
            result += ids

    return result


if __name__ == "__main__":
    conn = Connection()
    collection = conn.get_english_collection()

    adv = BaseTextPreprocessor(filename="training_data\\ad.txt", collection=collection, category='ADVERT')
    adv.process_data()
    adv_features = adv.get_most_frequent(50)
    adv_tokens = adv.category_tokens
    adv_tweets = adv.tweets_lemmas
    print('adv')
    print(adv_tweets)
    print(adv_features)

    pos_neg_ids = get_training_ids_ms("training_data\\positive.txt", "training_data\\negative.txt")
    print('len of pos and neg list', len(pos_neg_ids), pos_neg_ids[0])
    not_adv = BaseTextPreprocessor(ids_list=pos_neg_ids, collection=collection,
                                   category='NOT_AD')
    not_adv.process_data()
    not_adv_features = not_adv.get_most_frequent(50)
    not_adv_tokens = not_adv.category_tokens
    not_adv_tweets = not_adv.tweets_lemmas
    print('not adv')
    print(not_adv_tweets)
    print(not_adv_features)

    documents = combine_and_shuffle(adv_tweets, not_adv_tweets)
    word_features = combine_and_shuffle(adv_tokens, not_adv_tokens)
    # print(documents[0])

    featuresets = []
    for tweet, category in documents:
        featuresets.append((find_features(tweet, word_features), category))
    print('featureset', featuresets)

    training_set = featuresets[:900]
    testing_set = featuresets[900:]

    print(training_set[0])
    print(testing_set[0])

    # NAIVE BAYES CLASSIFIER
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original NB classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

    classifier.show_most_informative_features(20)

    # LinearSVC_classifier = SklearnClassifier(LinearSVC())
    # LinearSVC_classifier.train(testing_set)
    # print("Linear SVC classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,
    #                                                                          testing_set)) * 100)

    # pickle classifier
    serialize(classifier, "pickled_objects\\adv_vs_notadv_NBC.pickle")

    # pickle word features
    serialize(word_features, "pickled_objects\\adv_vs_notadv_word_features.pickle")

    # pickle featuresets
    serialize(featuresets, "pickled_objects\\adv_vs_notadv_featuresets.pickle")

    conn.close_connection()
