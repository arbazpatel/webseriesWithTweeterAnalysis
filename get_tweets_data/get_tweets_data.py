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
    # tweets = tweets
    # print(tweets)
 

#     file_name = 'Sentiment_Analysis_of_{}_Tweets_About_{}.csv'.format(30, ht)
    
#     with open(file_name, 'w', newline='') as csvfile:
#         csv_writer = csv.DictWriter(
#             f=csvfile,fieldnames=["Series", "Positive","Negative", "Neutral"])
#         csv_writer.writeheader()
    
#     for singletweet in tweets:
#         Result =  TextBlob(singletweet.text)
#         Polarity = Result.sentiment.polarity
#         print(Polarity)
#         if Polarity > 0:
#             positive_list.append([Polarity, ht])

#         elif Polarity <0:
#             negative_list.append([Polarity, ht])
#         else:
#             neutral_list.append([Polarity,ht])

# print(positive_list, 'pos list')
# print(negative_list, 'neg list')
# print(neutral_list, 'neutral list')

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
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
        text = re.sub(regex_pattern,'',text)

        # The below block of code removes urls from the text.
        pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
        text = re.sub(pattern,'',text)

        # The following block removes @mentions and hashes from the text.
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

        # scoreFileWriter.writerow([tweet.text,polarity])

        # scoreFileWriter.writerow({
        #     tweet.text,
        # polarity
        # })
 
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
    # print(positive, negative, neutral)
    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')

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




# with open(file_name,'w') as scorefile:
#     scoreFileWriter = csv.writer(scorefile)
#     scoreFileWriter.writerow(["Series", "Positive","Negative", "Neutral"])
#     scoreFileWriter.writerow([
#         ht,
#         positive,
#         negative,
#         neutral
#     ])

# tweet_list = pd.DataFrame(tweet_list)
# neutral_list = pd.DataFrame(neutral_list)
# negative_list = pd.DataFrame(negative_list)
# positive_list = pd.DataFrame(positive_list)
# print("total number: ",len(tweet_list))
# print("positive number: ",len(positive_list))
# print("negative number: ", len(negative_list))
# print("neutral number: ",len(neutral_list))


# labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
# sizes = [positive, neutral, negative]
# colors = ['yellowgreen', 'blue','red']
# patches, texts = plt.pie(sizes,colors=colors, startangle=90)
# plt.style.use('default')
# plt.legend(labels)
# plt.title("Sentiment Analysis Result for keyword= "+keyword+"" )
# plt.axis('equal')
# plt.show()

# tweet_list.drop_duplicates(inplace = True)



# def removeDuplicates():
#     return tweet_list.drop_duplicates(inplace = True)

# def cleanTweets(tweet_list):
#     tw_list = pd.DataFrame(tweet_list)
#     tw_list["text"] = tw_list[0]
#     #Removing RT, Punctuation etc
#     remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
#     rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
#     tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
#     tw_list["text"] = tw_list.text.str.lower()
#     tw_list.head(10)

# tw_list = pd.DataFrame(tweet_list)
# tw_list["text"] = tw_list[0]

#Removing RT, Punctuation etc
# remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
# rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
# tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
# tw_list["text"] = tw_list.text.str.lower()
# tw_list.head(10)


# tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
# for index, row in tw_list['text'].iteritems():
#     score = SentimentIntensityAnalyzer().polarity_scores(row)
#     neg = score['neg']
#     neu = score['neu']
#     pos = score['pos']
#     comp = score['compound']
#     if neg > pos:
#         tw_list.loc[index, 'sentiment'] = "negative"
#     elif pos > neg:
#         tw_list.loc[index, 'sentiment'] = "positive"
#     else:
#         tw_list.loc[index, 'sentiment'] = "neutral"
    
#     tw_list.loc[index, 'neg'] = neg
#     tw_list.loc[index, 'neu'] = neu
#     tw_list.loc[index, 'pos'] = pos
#     tw_list.loc[index, 'compound'] = comp
# print(tw_list.head(10))

# tw_list.to_csv('file1.csv')

# Creating new data frames for all sentiments (positive, negative and neutral)tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
# tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
# tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]


# def count_values_in_column(data,feature):
#     total=data.loc[:,feature].value_counts(dropna=False)
#     percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
#     return pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])

# #Count_values for sentiment
# count_values_in_column(tw_list,"sentiment")

# def pieChartForSentiments():
#     # create data for Pie Chart
#     pichart = count_values_in_column(tw_list,"sentiment")
#     names= pichart.index
#     size=pichart["Percentage"]
 
#     # Create a circle for the center of the plot
#     my_circle=plt.Circle( (0,0), 0.7, color='white')
#     plt.pie(size, labels=names, colors=['green','blue','red'])
#     p=plt.gcf()
#     p.gca().add_artist(my_circle)
#     plt.show()


#Function to Create Wordcloud
# def create_wordcloud(text):
#     mask = np.array(Image.open("cloud.png"))
#     stopwords = set(STOPWORDS)
#     wc = WordCloud(background_color="white",
#     mask = mask,
#     max_words=3000,
#     stopwords=stopwords,
#     repeat=True)
#     wc.generate(str(text))
#     wc.to_file("wc.png")
#     print("Word Cloud Saved Successfully")
#     path="wc.png"
#     display(Image.open(path))

# create_wordcloud(tw_list["text"].values)

#Creating wordcloud for positive sentiment
# create_wordcloud(tw_list_positive["text"].values)

#Creating wordcloud for negative sentiment
# create_wordcloud(tw_list_negative["text"].values)

#Calculating tw'et's lenght and word count
# tw_list['text_len'] = tw_list['text'].asty

# pe(str).apply(len)
# tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

# round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)

#Removing Punctuation
# def remove_punct(text):
#     text = "".join([char for char in text if char not in string.punctuation])
#     text = re.sub('[0-9]+', '', text)
#     return text
 
# tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

# #Applying tokenization
# def tokenization(text):
#     text = re.split('\W+', text)
#     return text

# tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))#Removing stopwords
# stopword = nltk.corpus.stopwords.words('english')
# def remove_stopwords(text):
#     text = [word for word in text if word not in stopword]
#     return text
    
# tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))

# #Applying Stemmer
# ps = nltk.PorterStemmer()

# def stemming(text):
#     text = [ps.stem(word) for word in text]
#     return text

# tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

# #Cleaning Text
# def clean_text(text):
#     text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
#     text_rc = re.sub('[0-9]+', '', text_lc)
#     tokens = re.split('\W+', text_rc)    # tokenization
#     text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
#     return text
# tw_list.head()


#Appliyng Countvectorizer
# countVectorizer = CountVectorizer(analyzer=clean_text) 
# countVector = countVectorizer.fit_transform(tw_list['text'])
# print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
# #print(countVectorizer.get_feature_names())

# # xx Number of reviews has xx words
# count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
# count_vect_df.head()

# # sort values as a descending to see most used words
# # Most Used Words
# count = pd.DataFrame(count_vect_df.sum())
# countdf = count.sort_values(0,ascending=False).head(20)
# countdf[1:11]


"""
Building n gram model helps us to predict most probably word that might follow this sequence. Firstly let’s create a function then built n2_bigram, n3_trigram etc.
"""
#Function to ngram
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


#n2_bigram
# n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)
# print(n2_bigrams)

# #n3_trigram
# n3_trigrams = get_top_n_gram(tw_list['text'],(3,3),20)
# n3_trigrams
