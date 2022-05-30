#!/usr/bin/python
import tweepy
import csv #Import csv
consumer_key = 'RkQqpzW896xtPE3V6QW1RDJeK'
consumer_secret = 'PvUXFfbl8NdOyTjJj5fR0ScoFaHYTAEShRncmBRcIkJOc7yMy2'
access_token = '939822906564947973-saYHbiZPbgXwJuPelYTM7SJ5hP7R7zS'
access_token_secret = 'vdHgGxylQWLjZ5ee2S20YjLHNElwtutNnTUVtXNwbqU4o'
India_WOE_ID = 1
auth = tweepy.OAuthHandler(consumer_key= consumer_key, consumer_secret= consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# Open/create a file to append data to
csvFile = open('../raw_twwets/raw_tweets.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search_tweets, q = "NetFlix Webseries"+ " -filter:retweets", lang = "en").items(40):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    # print(tweet.created_at, tweet.text)
csvFile.close()
print("__________________GENERATED CSV___________________")