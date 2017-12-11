import nltk
import os
import json


DATA_DIRNAME = "\\data\\"
CUR_DIRNAME = os.path.dirname(os.path.abspath(__file__))

data_filenames = os.listdir(CUR_DIRNAME + DATA_DIRNAME)

counter = 0
en_c = 0
ru_c = 0

processed_by_id = []

for filename in data_filenames[:20]:
    print('\n', filename, '\n')
    with open(CUR_DIRNAME + DATA_DIRNAME + filename, 'r') as f:
        raw_data = f.read()
        data = json.loads(raw_data)
        counter += 1
        print('\n', counter, '\n')

        for tweet in data:
            if tweet['id'] in processed_by_id:
                continue
            #if len(tweet['entities']['urls']) > 0:
            if 'http' in tweet['text']:
                continue

            if tweet['lang'] == 'en':
                en_c += 1
                print('>>>', tweet['text'])
            elif tweet['lang'] == 'ru':
                ru_c += 1
                print('>>>', tweet['text'])

            processed_by_id.append(tweet['id'])

print('English tweets amount: ', en_c)
print('Russian tweets amount: ', ru_c)