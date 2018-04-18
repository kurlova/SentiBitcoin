import tweepy
from time import sleep

from db_manager import db_insert_data
from config import *


class TwitterScraper():
    """
    Collects data from Twitter Search API
    """
    def __init__(self, query):
        self.tweets = []
        self.total_count = 0
        self.query = query
        self.results_per_page = 100
        self.max_id = None
        self.minutes_to_sleep = 20
        self.CREDENTIALS = (ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    def authenticate(self):
        """
        Authenticates using Tweepy OAuth with a given credentials
        """
        try:
            auth = tweepy.OAuthHandler(self.CREDENTIALS[2], self.CREDENTIALS[3])
            auth.set_access_token(self.CREDENTIALS[0], self.CREDENTIALS[1])
            api = tweepy.API(auth)
            return api
        except:
            print('Could not authenticate with given credentials.')
            return None

    def collect_tweets(self, api):
        """
        Collects all tweets that can be reached at a time for a given query
        """
        print('coll tw')
        for tweet in tweepy.Cursor(api.search,
                                   q=self.query,
                                   rpp=self.results_per_page,
                                   max_id=self.max_id,
                                   tweet_mode="extended"
                                   ).items():
            print(tweet.created_at)
            self.tweets.append(tweet._json)

    def write_to_db(self, data):
        db_insert_data(data)
        print(data[-1])
        print("data inserted")

    def manage_weekly_scraping(self):
        """
        Creates and manages a process of scraping weekly data
        """
        while True:
            api = self.authenticate()

            if not api:
                return

            self.tweets = []
            try:
                self.collect_tweets(api)
            except Exception as e:
                self.change_credentials()
                if e.response.status_code == 429:
                    print("Rate of collecting tweets per 15 minutes is limited")
                    self.write_to_db(self.tweets)

                    print('Waiting for 20 minutes...')
                    sleep(60 * self.minutes_to_sleep)
                    continue
                else:
                    print('Collecting tweets stopped for unknown reason')
                    return


    def change_credentials(self):
        if ACCESS_TOKEN in self.CREDENTIALS:
            self.CREDENTIALS = (ACCESS_TOKEN2, ACCESS_SECRET2, CONSUMER_KEY2, CONSUMER_SECRET2)
        else:
            self.CREDENTIALS = (ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

        print(self.CREDENTIALS)


if __name__ == "__main__":
    words_for_search = 'Bitcoin'

    scraper = TwitterScraper(words_for_search)

    try:
        scraper.manage_weekly_scraping()
    except KeyboardInterrupt:
        print('Done')
