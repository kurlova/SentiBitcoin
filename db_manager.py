import datetime

from pymongo import MongoClient


def db_insert_data(data):
    client = MongoClient()
    db = client['data']
    collection_en = db['tweets_en']
    collection_ru = db['tweets_ru']

    for tweet in data:
        if tweet['lang'] == 'en':
            collection_en.insert_one(tweet)
        elif tweet['lang'] == 'ru':
            collection_ru.insert_one(tweet)
        else:
            continue
    client.close()


class Connection:
    """
    Represents a connection to MongoDB database
    """
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client['data']
        self.collection_en = self.db['tweets_en']

    def get_client(self):
        return self.client

    def get_db(self):
        return self.db

    def get_english_collection(self):
        return self.collection_en

    def close_connection(self):
        self.client.close()
        print('Connection closed')

    def get_tweets(self):
        """
        Get tweets for today
        :return: list with tweets
        """
        # if language == "en":
        collection = self.collection_en
        today = datetime.datetime.now() - datetime.timedelta(days=1)
        today_begin = today.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%a %b %d %H:%M:%S %z %Y')
        today_end = today.replace(hour=23, minute=59, second=59).strftime('%a %b %d %H:%M:%S %z %Y')
        tweets = list(collection.find({'created_at': {'$gt': today_begin, '$lt': today_end},
                                       'entities.urls': {'$size': 0},  # no urls allowed
                                       'retweeted': {'$eq': False},    # must not be a retweet
                                       }))
        print(f'{len(tweets)} tweets for today (since {today_begin}) are found.')

        return tweets
