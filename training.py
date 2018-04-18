# {retweet_count: {$eq: 0},'entities.urls': {$size: 0},truncated: {$eq: false}}     {_id: -1}

import nltk
import traceback
from pymongo import MongoClient
from random import shuffle

# check for wordnet presence, download if absent
from nltk.stem import WordNetLemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize('test')
except:
    nltk.download('wordnet')

# check for POS tagger presence, download if absent
from nltk import pos_tag
try:
    pos_tag('test text.')
except:
    nltk.download('averaged_perceptron_tagger')

# check for word tokenizer presence, download if absent
from nltk.tokenize import word_tokenize
try:
    word_tokenize('test text.')
except:
    nltk.download('punkt')

# check for english stopwords corpus presence, download if absent
from nltk.corpus import stopwords
try:
    stopwords.words("english")
except:
    nltk.download('stopwords')


STOPWORDS_ENG = set(stopwords.words("english"))


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


class BaseTextPreprocessor:
    """
    Provides methods for data processing for further classification
    """
    def __init__(self, filename, collection):
        self.filename = filename
        self.lemmatizer = WordNetLemmatizer()
        self.collection = collection
        self.category_tokens = []

    def get_training_ids(self):
        """
        Get IDs of training data in database

        :param filename: file from which to extract data
        :return: list with IDs
        """
        with open(f"training_data\\{self.filename}", "r") as f:
            ids = f.readlines()
            ids = [tweet_id.rstrip('\n') for tweet_id in ids]

        return ids

    def get_text(self, tweet):
        """
        Find "full_text", if not present, find "text"
        Not all tweets have full_text option, unfortunately :(

        :param tweet: json object representing a tweet
        :return: tweet['text'] or tweet['full_text']
        """
        try:
            text = tweet["full_text"]
        except:
            text = tweet["text"]

        return text

    def tokenize_by_sentence(self, tweet):
        """
        Turn tweet into a set of sentences

        :param tweet: a single tweet
        :return: a set of sentences in the tweet
        """
        return nltk.sent_tokenize(tweet)

    def tokenize_by_word(self, tweet):
        """
        Turn tweet into set of tokens
        Transforms each token to lowercase

        :param tweet: a single string tweet
        :return: a set of words of the tweet. Example: ['this', ' ', 'is', ' ', 'a', ' ', 'tweet']
        """
        return list(map(lambda token: token.lower(), word_tokenize(tweet)))

    def remove_stop_words(self, tokens):
        """
        Filters english stop words from tokenized tweet.
        Examples of stop words: a, the, me, you

        :param tokens: list of words of a tweet. Example: ['this', ' ', 'is', ' ', 'a', ' ', 'tweet']
        :return: filtered list that does not include stopwords
        """
        return list(filter(lambda token: token not in STOPWORDS_ENG, tokens))

    def get_general_words(self):
        """
        Get words that are common for all categories, such as 'bitcoin', 'coin', 'btc', '#' etc.
        These words should not influence the result

        :return: a list with general common words
        """
        filename = 'common_words.txt'

        with open(filename, 'r') as f:
            data = f.readlines()
            data = [word.rstrip('\n') for word in data]

        return data

    def remove_general_words(self, tokens):
        """
        Removes common words such as 'bitcoin' from tweet tokens

        :param tokens: tweet tokens
        :return: tokens without common tokens
        """
        general_words = self.get_general_words()

        return list(filter(lambda x: x not in general_words, tokens))

    def remove_emoticons(self, tweet_tokens):
        """
        Removes emoticons from token
        This method can be actually developed into emoticon analysis,
        as emoticons are also interpretaions of emotions, thus can be analysed

        :param tweet_tokens: list with tweet tokens
        :return: list of tokens with replaced emoticons
        """
        tweet_tokens = list(map(lambda x: "".join(i for i in x if ord(i) < 128), tweet_tokens))
        tweet_tokens = list(filter(lambda x: x != '', tweet_tokens))

        return tweet_tokens

    def left_significant_words(self, tweet_tokens):
        """
        Removes stop words
        Removes common words
        Removes emoticons

        :param tweet_tokens: list with tweet tokens
        :return: filtered list of tokens
        """
        no_stop_words = self.remove_stop_words(tweet_tokens)
        # print('no stop words', no_stop_words)
        no_general_words = self.remove_general_words(no_stop_words)
        # print('no general words', no_general_words)
        no_emoticons = self.remove_emoticons(no_general_words)
        # print('no emoticons', no_emoticons)

        return no_emoticons

    def get_part_of_speech(self, tokens):
        """
        Defines what part of speech the word is
        See full list of POS in 'POS abbreviations.txt' file

        :param tokens: word
        :return: tuple (word, POS abbreviation)
        """
        pos_tokens = pos_tag(tokens)

        return pos_tokens

    def lemmatize(self, token):
        """
        Turns a word (token) into its lemma

        :param token: word to lemmatize
        :return: lemma of the token
        """
        return self.lemmatizer.lemmatize(token)

    def get_most_frequent(self, n):
        """
        Calculates n most frequent words on a base of self.category_token

        :param n: desired number of most frequent words
        :return: a list with n most frequent words
        """
        return nltk.FreqDist(self.category_tokens).most_common(n)

    def process_data(self):
        """
        Data processing management

        :return:
        """
        training_ids = self.get_training_ids()

        for tweet_id in training_ids:
            # print(tweet_id)
            tweet = self.collection.find_one({"id": int(tweet_id)})
            tweet_text = self.get_text(tweet)
            # print(tweet_text)
            tweet_sentences = self.tokenize_by_sentence(tweet_text)
            # print('sentence tokenizer', tweet_sentences)

            for sentence in tweet_sentences:
                # print(sentence)
                tweet_tokens_all = self.tokenize_by_word(sentence)
                # print('tokens', tweet_tokens_all)
                tweet_tokens = self.left_significant_words(tweet_tokens_all)
                pos_tokens = self.get_part_of_speech(tweet_tokens)
                # print(pos_tokens)

                for token in pos_tokens:
                    lemma = self.lemmatize(token[0])
                    # print('lemma of', token[0], ':', lemma)

                    self.category_tokens.append(lemma)
            # print()
        print(len(self.category_tokens))


def combine_and_shuffle(*args):
    """
    Randomize order of tweet tokens, so negative will be among positive
    (not the whole negative group after whole positive group)
    :return: shuffled list of joined together multiple lists
    """
    res = []

    for datalist in args:
        res += datalist

    shuffle(res)

    return res


def find_features(document, word_features):
    """
    Checks id a words from word_features exists in document

    :param document: positive / negative tokens
    :param word_features: common words from both negative and positive
    :return:
    """
    words = set(document)
    features = {}

    for wf in word_features:
        features[wf] = wf in words

    return features


if __name__ == "__main__":
    # set up connection
    conn = Connection()
    collection = conn.get_english_collection()

    try:
        # work with positive tweets
        positive = BaseTextPreprocessor(filename="positive.txt", collection=collection)
        positive.process_data()
        positive_features = positive.get_most_frequent(100)
        print(positive_features)
        positive_tokens = positive.category_tokens
        print(f'len of positive tokens: {len(positive_tokens)}')

        # work with negative tweets
        negative = BaseTextPreprocessor(filename="negative.txt", collection=collection)
        negative.process_data()
        negative_features = negative.get_most_frequent(100)
        print(negative_features)
        negative_tokens = negative.category_tokens
        print(f'len of negative tokens: {len(negative_tokens)}')

        word_features = combine_and_shuffle(positive_tokens, negative_tokens)
        print(f'common length: {len(word_features)}')

        # see which words are common for positive, and which are common for negative
        positive_in_features = find_features(positive_tokens, word_features)
        negative_in_features = find_features(negative_tokens, word_features)
        print('positive', positive_in_features)
        print('negative', negative_in_features)
    except:
        traceback.print_exc()

    # close connection
    conn.close_connection()
    print('Connection closed')
