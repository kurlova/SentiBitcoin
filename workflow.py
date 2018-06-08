from data_processor import TextPreprocessor, find_features
from utils import deserialize, PCKL_POSNEG_FILENAME_START, PCKL_DATADIR, PCKL_ADNOTAD_FILENAME_START, \
    PCKL_OBJSUBJ_FILENAME_START
from db_manager import Connection


if __name__ == "__main__":
    conn = Connection()
    collection = conn.get_english_collection()
    word_features_pn = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "word_features"))
    NBC_classifier_pn = deserialize(PCKL_DATADIR.format(PCKL_POSNEG_FILENAME_START + "NBC_classifier"))
    NBC_classifier_advnotadv = deserialize(PCKL_DATADIR.format(PCKL_ADNOTAD_FILENAME_START + "NBC_classifier"))
    word_features_advnotadv = deserialize(PCKL_DATADIR.format(PCKL_ADNOTAD_FILENAME_START + "word_features"))
    NBC_classifier_objsubj = deserialize(PCKL_DATADIR.format(PCKL_OBJSUBJ_FILENAME_START + "NBC_classifier"))
    word_features_objsubj = deserialize(PCKL_DATADIR.format(PCKL_OBJSUBJ_FILENAME_START + "word_features"))

    tweets = conn.get_tweets()[:200]

    for tweet in tweets:
        tweet_data = TextPreprocessor(tweet_json=tweet)
        tweet_data.process()
        tweet_lemmas = tweet_data.tweets_lemmas[0]
        tweet_features_objsubj = find_features(tweet_lemmas, word_features_objsubj)
        result_objsubj = NBC_classifier_objsubj.classify(tweet_features_objsubj)
        # print(result)

        if result_objsubj == "SUBJ":
            print(tweet_data.tweet_text)
            tweet_features_advnotadv = find_features(tweet_lemmas, word_features_advnotadv)
            result_advnotadv = NBC_classifier_advnotadv.classify(tweet_features_advnotadv)
            print(result_advnotadv)

            if result_advnotadv == "NOT_AD":
                tweet_features_pn = find_features(tweet_lemmas, word_features_pn)
                result_pn = NBC_classifier_pn.classify(tweet_features_pn)
                print(result_pn)
            print('---------------------------------\n')


    conn.close_connection()

