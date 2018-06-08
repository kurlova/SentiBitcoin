from nltk import SklearnClassifier
from sklearn.svm import LinearSVC

from data_processor import *
from utils import serialize, PCKL_DATADIR


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


def get_indexes_80_20(length):
    """
    Follows a rule of 80/20:
    80% of data for trainig, 20% of data for testing

    :param length: length of data
    :return: index for 80%, index for 20%
    """
    i_80 = length * 80 // 100
    i_20 = length - i_80
    return i_80, i_20


def train_classifier(collection, filename_start, **kwargs):
    main_cat_name = kwargs["main_cat_name"]
    main_cat_filenames = kwargs["main_cat_filenames"]
    opposite_cat_name = kwargs["opposite_cat_name"]
    opposite_cat_filenames = kwargs["opposite_cat_filenames"]

    main_cat_ids = get_training_ids_ms(*main_cat_filenames)
    main_cat = TrainTextPreprocessor(ids_list=main_cat_ids, db_collection=collection, category=main_cat_name)
    main_cat.process_data()
    # main_cat_features = main_cat.get_most_frequent(50)
    main_cat_tokens = main_cat.category_tokens
    main_cat_tweets = main_cat.tweets_lemmas_categorized

    opposite_cat_ids = get_training_ids_ms(*opposite_cat_filenames)
    opposite_cat = TrainTextPreprocessor(ids_list=opposite_cat_ids, db_collection=collection,
                                         category=opposite_cat_name)
    opposite_cat.process_data()
    # opposite_cat_features = opposite_cat.get_most_frequent(50)
    opposite_cat_tokens = opposite_cat.category_tokens
    opposite_cat_tweets = opposite_cat.tweets_lemmas_categorized
    # print(opposite_cat_tokens)

    # Compute TF-IDF
    corpus = main_cat.tweets_lemmas + opposite_cat.tweets_lemmas
    tf_idf_range = compute_tfidf(corpus)
    for el in tf_idf_range:
        # print(el)
        pass

    documents = combine_and_shuffle(main_cat_tweets, opposite_cat_tweets)
    word_features = combine_and_shuffle(main_cat_tokens, opposite_cat_tokens)

    featuresets = []
    for tweet, category in documents:
        featuresets.append((find_features(tweet, word_features), category))

    train_index, test_index = get_indexes_80_20(len(featuresets))
    training_set = featuresets[:train_index]
    testing_set = featuresets[test_index:]

    # NAIVE BAYES CLASSIFIER
    # NBC_classifier = nltk.NaiveBayesClassifier.train(training_set)
    # print("Original NB classifier accuracy percent:", (nltk.classify.accuracy(NBC_classifier, testing_set)) * 100)
    #
    # NBC_classifier.show_most_informative_features(50)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("Linear SVC classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,
                                                                             testing_set)) * 100)

    # serialize(documents, "..\\" + PCKL_DATADIR.format(filename_start + "documents"))
    # serialize(word_features, "..\\" + PCKL_DATADIR.format(filename_start + "word_features"))
    # serialize(featuresets, "..\\" + PCKL_DATADIR.format(filename_start + "featuresets"))
    # serialize(NBC_classifier, "..\\" + PCKL_DATADIR.format(filename_start + "NBC_classifier"))
