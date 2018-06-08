import json
from db_manager import Connection


def read_tweets(filename):
    with open(filename, 'r') as f:
        data = f.readlines()

    return data


conn = Connection()
tids = read_tweets('training_data\\news.txt')
conn.close_connection()

for tid in tids:
    print(tid)
    tid = tid.strip('\n')
    tweet = conn.collection_en.find({'id': int(tid)})

    try:
        text = tweet["full_text"]
    except:
        text = tweet["text"]

    print(text)
    print('-'*50)

