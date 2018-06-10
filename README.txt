1. We sampled 100 tweets. This is an arguably small data-set which does not give a wide/accurate representation of tweet sentiments. 
2. There's a wide distribution of tweet sentiments displayed on the scatterplot however there's a more positive leaning distribution. There is concentration of tweets displaying neutral sentiments meaning the a substantial portion of each tweet from each of the media sources had a compound score of zero.
3. The overall media sentiment bar-chart shows that overall, each media outlet expressed positive sentiments. NY Times and showed the least positive (just over .05) while CBS showed the most positive (over .30).

README

#dependencies
import pandas as pd
import tweepy
import time
import json
import random
from config import consumer_key, consumer_secret, access_token, access_token_secret
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt

# Twitter API Keys
consumer_key = consumer_key
consumer_secret = consumer_secret
access_token = access_token
access_token_secret = access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target User BBC, CBS, CNN, Fox, and New York times
target_user = ("BBC", "CBS", "CNN", "Fox", "nytimes")

#sentiment holder
sentiments = []

#loop through each term
for user in target_user:
    compound_list = []
    counter = 1
    
    #get feed
    for x in range(5):
        public_tweets = api.user_timeline('@' + user, page=x)
        
        for tweet in public_tweets:
            #vader!
            compound = analyzer.polarity_scores(tweet['text'])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            tweets_ago = counter


            
            sentiments.append({"User": user, 
                               "Date": tweet["created_at"],
                               "Tweet": tweet["text"],
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neu,
                               "Neutral": neg,
                                "Tweets Ago": counter})
            counter = counter + 1
            

#convert sentiments[] to df
sentiments_df = pd.DataFrame(sentiments,
                            columns=["User","Date","Tweet","Positive","Negative","Neutral","Tweets Ago","Compound"])
sentiments_df.to_csv('NewsTudes.csv')
sentiments_df.head()

sns.set_style("ticks")
plt.style.use("seaborn")


#scatterplot with seaborn
sns.lmplot(x ="Tweets Ago", 
           y ="Compound",
           data=sentiments_df, 
           hue="User",
           fit_reg=False,
           palette = "bright",
           size = 6,
           aspect = 1.5,
           scatter_kws={"marker": "D",
                        "s": 80,
                      "edgecolor":sns.xkcd_rgb["black"],
                      "linewidth": 1})
plt.title("Sentiment Analysis of News Org Tweets ({})".format(tweet["created_at"]), fontsize = 18, fontweight='bold')
plt.xlabel("Tweets Ago", labelpad=10, fontsize = 14)
plt.ylabel("Tweet Polarity",fontsize = 14)
plt.subplots_adjust(top=0.88)
plt.xticks(size = 12)
plt.yticks(size = 12)

# Save png
plt.savefig("Sentiment_Analysis_ScatterPlot.png")

plt.show()

# Group and calc overall compound score
overall_sentiment = sentiments_df.groupby(['User']).mean()["Compound"]
overall_sentiment_pd = pd.DataFrame.from_dict(overall_sentiment)
overall_sentiment_pd["Compound"]

# make bar chart with matplotlib
colors = ("#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400")
plt.bar(target_user, overall_sentiment_pd["Compound"], color=colors, alpha = 1, width =1, 
        edgecolor="black", linewidth=0.5)

#title, x and y labels
plt.title("Overall Media Sentiment based on Twitter ({})".format(tweet["created_at"]), fontsize = 13, 
          fontweight='bold')
plt.xlabel("Media Sources", labelpad=10, fontsize = 14)
plt.ylabel("Tweet Polarity",fontsize = 14)

plt.show()
# Save png
plt.savefig("Overall_Sentiment_Analysis_BarChart.png")