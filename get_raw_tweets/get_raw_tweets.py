#!/usr/bin/python
import tweepy
import csv 
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
token = WordPunctTokenizer()
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
# For authentication of tweepy API
consumer_key = 'RkQqpzW896xtPE3V6QW1RDJeK'
consumer_secret = 'PvUXFfbl8NdOyTjJj5fR0ScoFaHYTAEShRncmBRcIkJOc7yMy2'
access_token = '939822906564947973-saYHbiZPbgXwJuPelYTM7SJ5hP7R7zS'
access_token_secret = 'vdHgGxylQWLjZ5ee2S20YjLHNElwtutNnTUVtXNwbqU4o'

India_WOE_ID = 1

# To authenticate tweepy
auth = tweepy.OAuthHandler(consumer_key= consumer_key, consumer_secret= consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


# Open/create a file to append data to
csvFile = open('../raw_tweets/raw_tweets.csv', 'a')


df = pd.read_csv('../raw_tweets/raw_tweets.csv')

print(df.head())


def word_clouds(del_amp_text):
    customStopwords=list(STOPWORDS)+ ['the','an','pandamic','people','will','would', 'could','he','she']
 
    wordcloudimage = WordCloud(
                          max_words=100,
                          max_font_size=500,
                          font_step=2,
                          stopwords=customStopwords,
                          background_color='white',
                          width=1000,
                          height=720
                          ).generate(del_amp_text)
 
    plt.figure(figsize=(15,7))
    plt.axis("off")
    plt.imshow(wordcloudimage)
    plt.show()

#Use csv writer
csvWriter = csv.writer(csvFile)
tweets = []
for tweet in tweepy.Cursor(api.search_tweets, q = "NetFlix Webseries"+ " -filter:retweets", lang = "en").items(10):
    
    tweet.text = tweet.text.replace('@','')
    tweet.text = tweet.text.replace('\xe2','')
    tweet.text = tweet.text.replace('\x80','')
    tweet.text = tweet.text.replace('\x98','')

    tweet.text = tweet.text.replace('\n','')
    tweet.text = tweet.text.replace('|','')

    # The following pattern will remove most emoticons and symbols in the text.
    regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    tweet.text = re.sub(regex_pattern,'',tweet.text)

    # The below block of code removes urls from the text.
    pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
    tweet.text = re.sub(pattern,'',tweet.text)

    # The following block removes @mentions and hashes from the text.
    re_list = ['@[A-Za-z0-9_]+', '#']
    combined_re = re.compile( '|'.join( re_list) )
    tweet.text = re.sub(combined_re,'',tweet.text)

    # The block below will remove html characters from the text.
    del_amp = BeautifulSoup(tweet.text, 'lxml')
    del_amp_text = del_amp.get_text()
    # print(del_amp_text)
    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, del_amp_text])
    tweets.append(del_amp_text)

csvFile.close()

# Combining list values to text and sending as a input to the function.

word_clouds(' '.join(str(e) for e in tweets))

print("__________________GENERATED CSV AND WORDCLOUD___________________")
