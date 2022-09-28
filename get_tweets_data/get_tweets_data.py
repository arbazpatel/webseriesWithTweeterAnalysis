from math import ceil
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string 
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import csv
import preprocessor as p

consumer_key = 'RkQqpzW896xtPE3V6QW1RDJeK'
consumer_secret = 'PvUXFfbl8NdOyTjJj5fR0ScoFaHYTAEShRncmBRcIkJOc7yMy2'
access_token = '939822906564947973-saYHbiZPbgXwJuPelYTM7SJ5hP7R7zS'
access_token_secret = 'vdHgGxylQWLjZ5ee2S20YjLHNElwtutNnTUVtXNwbqU4o'

India_WOE_ID = 1

auth = tweepy.OAuthHandler(consumer_key= consumer_key, consumer_secret= consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


#Sentiment Analysis
def percentage(part,whole):
    return int(round(100 * float(part)/float(whole),0))

# keyword = input("Please enter keyword or hashtag to search: ")
# noOfTweet = int(input ("Please enter how many tweets to analyze: "))
# tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
tweets_list = []

# hash_tags = ["Rudra","Family Man", "Red notice"]
hash_tags = pd.read_csv("../series_names_data/netflix_titles.csv", encoding="utf-8")

hash_tags_title = hash_tags['title'][:10] # This([::1000]) is for getting no. of records to fetch from csv file
print(hash_tags_title, 'all hash tags')
for index, ht in enumerate(hash_tags_title):
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    print(ht, 'tags')
    tweets = tweepy.Cursor(api.search_tweets, ht + " -filter:retweets").items(30)

    for tweet in tweets:
        # print(tweet.text)
        text = tweet.text.replace("&amp;","&").replace(",","").replace("RT","")
        text = p.clean(text)
        text = text.replace('@','')
        text = text.replace('\n','')
        text = text.replace('@','')
        text = text.replace('\xe2','')
        text = text.replace('\x80','')
        text = text.replace('\x98','')

        text = text.replace('\n','')
        text = text.replace('|','')

        regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (for iOS)
                           "]+", flags = re.UNICODE)
        text = re.sub(regex_pattern,'',text)

        # The below block of code removes urls from the text.
        pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
        text = re.sub(pattern,'',text)

        # The following block removes @ mentions and hashes from the text.
        re_list = ['@[A-Za-z0-9_]+', '#']
        combined_re = re.compile( '|'.join( re_list) )
        text = re.sub(combined_re,'',text)
    
        # The block below will remove html characters from the text.
        del_amp = BeautifulSoup(text, 'lxml')
        text = del_amp.get_text()


        tweet_list.append(text)
        analysis = TextBlob(text)
        score = SentimentIntensityAnalyzer().polarity_scores(text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity

        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
 
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1
    positive = percentage(positive, 30)
    negative = percentage(negative, 30)
    neutral = percentage(neutral, 30)
    polarity = percentage(polarity, 30)
    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')
    # print(positive, negative, neutral)

    if (positive >= negative) and (positive >= neutral):
        
        # print('Go and Watch',ht)
        what_to_do = 'Go and Watch'
        status = 'Positive'
    elif (negative >= positive) and (negative >= neutral):
        
        # print('Do not Watch', ht)
        what_to_do = 'Do not Watch'
        status = 'Negative'
    else:
        what_to_do = 'You Decide'
        status = "Neutral"
        # print('You Decide', ht)

    avg_rating = ((hash_tags['averageRating'][:10][index]+hash_tags['rotten_Tomatoes'][:10][index])/3)

    print(avg_rating, 'average ratings')
    if round(avg_rating)>5:
        avg_rating = 5.0

    tweets_list.append({'Web Series':ht, "Positive":positive, "Negative":negative, "Neutral":neutral, "Analysis":what_to_do, "Status":status, "Calculated_rating":avg_rating})

# import numpy as np

# np.savetxt("shows.csv", tweets_list, delimiter=", ", fmt="% s")
# print(tweets_list, 'tweet pos neg neu')
columns = ["Web Series", "Positive", "Negative", "Neutral","Analysis", "Status", "Calculated_rating"]
try:
    with open("tweet_analysis.csv", 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for key in tweets_list:
            writer.writerow(key)
except IOError:
    print("I/O error")
