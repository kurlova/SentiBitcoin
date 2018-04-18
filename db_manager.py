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


if __name__ == "__main__":
    pass
