from data_processor import TextPreprocessor, find_features
from utils import deserialize, PCKL_POSNEG_FILENAME_START, PCKL_DATADIR
from db_manager import Connection


if __name__ == "__main__":
    conn = Connection()
    collection = conn.get_english_collection()
    documents_ = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "documents"))
    word_features_ = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "word_features"))
    featuresets_ = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "featuresets"))
    NBC_classifier_ = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "NBC_classifier"))

    tweets = conn.get_tweets()[:20]
    for tweet_data in tweets:
        tweet_text = tweet_data["full_text"]
        print(tweet_text)
        tweet = TextPreprocessor(raw_tweet=tweet_text)
        tweet.process()
        tweet_lemmas = tweet.tweets_lemmas[0]
        print(tweet_lemmas)
        test_tweet_features = find_features(tweet_lemmas, word_features_)
        print(NBC_classifier_.classify(test_tweet_features))
        print('---------------------------------\n')

    conn.close_connection()

