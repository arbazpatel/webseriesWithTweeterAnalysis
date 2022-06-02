import pandas as pd
import numpy as np
import csv
hash_tags = pd.read_csv("../series_names_data/netflix_titles.csv", encoding="utf-8")
tweets_list=[]
hash_tags_title = hash_tags['title'][:1500] # This([::1000]) is for getting no. of records to fetch from csv file
# print(hash_tags_title, 'all hash tags')
for index, ht in enumerate(hash_tags_title):
    avg_rating = ((hash_tags['averageRating'][:1500][index]+hash_tags['rotten_Tomatoes'][:1500][index])/3)


    if round(avg_rating,1)>5:
        avg_rating = 5.0
    else:
        avg_rating = str(round(avg_rating, 1))

    tweets_list.append({"Calculated_rating":avg_rating})
# print(tweets_list)
# import numpy as np

# np.savetxt("shows.csv", tweets_list, delimiter=", ", fmt="% s")
# print(tweets_list, 'tweet pos neg neu')
columns = ["Calculated_rating"]
try:
    with open("tweet_analysis.csv", 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for key in tweets_list:
            writer.writerow(key)
except IOError:
    print("I/O error")

# print((((numVotes+averageRating+rotten_Tomatoes)/3)/1500), 'mean')