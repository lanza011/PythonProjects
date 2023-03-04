#!/usr/bin/env python
# coding: utf-8

# # DATA COLLECTION

# In[1]:


get_ipython().system('pip install --upgrade pip')


# In[2]:


get_ipython().system('pip3 install twint')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install -U textblob')
get_ipython().system('pip install emoji --upgrade')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install nest_asyncio')
get_ipython().system('pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib')


# In[3]:


import nest_asyncio
nest_asyncio.apply()
import pandas as pd
import tweepy
import json
import random
import os
from tqdm import tqdm, notebook
import numpy as np


# In[4]:


#APPLY FOR TWITTER API AND USE YOUR KEY AND TOKENS BELOW
#Part-1: Authorization and Search tweets
#Getting authorization
consumer_key = 'sQkxeidHCAdvyeIbdBNDev209'
consumer_secret = 'dK52I5fIjzGeZIWrnQLtAuYbfCOFEbURjZ5IqrjMup7A7y4rYh'
access_token = '1268998619077545984-DWZVle4u0WeFnxcqrEdhD8J4FGvuFq'
access_token_secret = '8dLWdzl2K6HYNa9w28xVoPYemTemSLYNwVYXGkf3J8k28'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=False)


# In[5]:


search_words = "WFH-filter:retweets"
date_since = "2020-01-01"

#Collect tweets
tweets = tweepy.Cursor(api.search_tweets,
              q=search_words,
              lang="en",
              since_id=date_since).items(1000)
#CHANGE THE ITEM(900) ABOVE TO THE NUMBER OF TWEETS THAT YOU WANT TO COLLECT 


# In[6]:


tweets_text = []


# In[7]:


tweets_copy = []
for tweet in tqdm(tweets):
     tweets_copy.append(tweet)


# In[8]:


tweets_df = pd.DataFrame()
for tweet in tqdm(tweets_copy):
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,
                                               'user_description': tweet.user.description,
                                               'user_created': tweet.user.created_at,
                                               'user_followers': tweet.user.followers_count,
                                               'user_friends': tweet.user.friends_count,
                                               'user_favourites': tweet.user.favourites_count,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               'source': tweet.source,
                                               'is_retweet': tweet.retweeted}, index=[0]))


# In[9]:


tweets_df.head()


# In[10]:


tweets_df.to_csv('twitter-WFH.csv', index=False, header=False)


# In[11]:


# Scrape Or Download Comments Using Python Through The Youtube Data API
# Watch the youtube video for explaination
# https://youtu.be/B9uCX2s7y7A

api_key = "AIzaSyBl5pbDfWCLRTQHB2Q_dRrvLUOAVT9BVv0" # Replace this dummy api key with your own.

from apiclient.discovery import build
youtube = build('youtube', 'v3', developerKey=api_key)

import pandas as pd

ID = "x6fIseKzzH0" # Replace this YouTube video ID with your own.

box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]


def scrape_comments_with_replies():
    data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults='100', textFormat="plainText").execute()

    for i in data["items"]:

        name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
        comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
        published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
        likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
        replies = i["snippet"]['totalReplyCount']

        box.append([name, comment, published_at, likes, replies])

        totalReplyCount = i["snippet"]['totalReplyCount']

        if totalReplyCount > 0:

            parent = i["snippet"]['topLevelComment']["id"]

            data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                            textFormat="plainText").execute()

            for i in data2["items"]:
                name = i["snippet"]["authorDisplayName"]
                comment = i["snippet"]["textDisplay"]
                published_at = i["snippet"]['publishedAt']
                likes = i["snippet"]['likeCount']
                replies = ""

                box.append([name, comment, published_at, likes, replies])

    while ("nextPageToken" in data):

        data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=data["nextPageToken"],
                                             maxResults='100', textFormat="plainText").execute()

        for i in data["items"]:
            name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
            comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
            published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
            likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
            replies = i["snippet"]['totalReplyCount']

            box.append([name, comment, published_at, likes, replies])

            totalReplyCount = i["snippet"]['totalReplyCount']

            if totalReplyCount > 0:

                parent = i["snippet"]['topLevelComment']["id"]

                data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                textFormat="plainText").execute()

                for i in data2["items"]:
                    name = i["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["textDisplay"]
                    published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ''

                    box.append([name, comment, published_at, likes, replies])

    df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box],
                       'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})

    df.to_csv('youtube-wfh.csv', index=False, header=False)

    return "Successful! Check the CSV file that you have just created."

scrape_comments_with_replies()


# In[12]:


import pandas as pd
pd.options.display.max_rows = 10

WFHT_df = tweets_df # pd.read_csv('twitter-AirCanada.csv')

WFHY_df = pd.read_csv('youtube-wfh.csv', index_col=0)


# In[13]:


WFHT_df.head()


# In[14]:


WFHY_df.head()


# In[15]:


WFHT_df=WFHT_df[['text']]
WFHY_df=WFHY_df[['Comment']]


# In[16]:


WFHY_df.rename(columns = {'Comment':'text'}, inplace = True)


# In[17]:


## merge all dataframes
df_list = [WFHT_df,WFHY_df]
tweets_df = pd.concat(df_list)


# In[18]:


tweets_df.info() 


# In[19]:


#tweets_df.to_csv('AirCanadaCom.csv', index=False, header=False)


# In[20]:


tweets_df.drop_duplicates(subset = ["text"], inplace=True)
print(f"all tweets: {tweets_df.shape}")


# In[21]:


tweets_df.columns


# In[22]:


tweets_df.head()


# In[23]:


tweets_df.tail()


# In[24]:


tweets_df['text'].nunique()


# In[25]:


# load library
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download the set of stop words the first time
import nltk
nltk.download('stopwords')


# In[26]:


# Load stop words
stop_words = stopwords.words('english')

# Show stop words
stop_words[:10]


# In[27]:


#Cleaning Tweets
import re 
from textblob import TextBlob 
import emoji  

def clean_tweet(text): 
    text = re.sub(r'@[A-Za-z0-9]+', '', str(text)) # remove @mentions
    text = re.sub(r'#', '',  str(text)) # remove the '#' symbol
    text = re.sub(r'RT[\s]+', '',  str(text)) # remove RT
    text = re.sub(r'https?\/\/S+', '',  str(text)) # remove the hyperlink
    text = re.sub(r'http\S+', '',  str(text)) # remove the hyperlink
    text = re.sub(r'www\S+', '',  str(text)) # remove the www
    text = re.sub(r'twitter+', '',  str(text)) # remove the twiiter
    text = re.sub(r'pic+', '',  str(text)) # remove the pic
    text = re.sub(r'com', '',  str(text)) # remove the pic

    return text

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


# In[28]:


tweets_df['cleaned_text']=tweets_df['text'].apply(clean_tweet)
#tweets_df['cleaned_text']=tweets_df['cleaned_text'].apply(remove_emoji)


# In[29]:


# Remove stop words
tweets_df['cleaned_text']=tweets_df['cleaned_text'].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))


# In[30]:


tweets_df.head()


# # DATA CLEANING

# In[31]:


from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


# In[32]:


tweets = tweets_df


# In[33]:


comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in tweets.cleaned_text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1000, height = 800, 
                background_color='white', colormap='Set2', 
                collocations=False, 
                stopwords = stopwords, 
                min_font_size = 12).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (10, 10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[34]:


#tweets.to_csv("DataOne.csv")


# In[35]:


##Merge all the data scraped from every week
#Change the names of the CSV Files to the keywords you used for each search to make it easy for located your dataset

#Tweets1 = pd.read_csv("DataOne.csv")
#Tweets2 = pd.read_csv("DataTwo.csv")
#Tweets3 = pd.read_csv("DataThree.csv")


# In[36]:


## merge all dataframes
#df_list = [Tweets1,Tweets2,Tweets3]
#df = pd.concat(df_list)


# In[37]:


tweets_df.head()


# In[38]:


##Save your combined dataset

#df.to_csv("Combined_Dataset.csv")


# In[39]:


#df=pd.read_csv("Combined_Dataset.csv")


# In[40]:


df = tweets_df


# # Data Analysis

# In[41]:


import pandas as pd
import numpy as np
import string
import re
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[42]:


def getSubjectivity(text):
    return TextBlob( str(text)).sentiment.subjectivity

def getPolarity(text):
    return TextBlob( str(text)).sentiment.polarity


# In[43]:


tweets.dropna(subset=['cleaned_text'], inplace = True)
tweets.reset_index(drop=True, inplace=True)


# In[44]:


tweets['Subjectivity'] = tweets['cleaned_text'].apply(getSubjectivity)
tweets['Polarity'] = tweets['cleaned_text'].apply(getPolarity)
tweets.head()


# In[45]:


# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def get_Polarity_Analysis(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'
def get_Subjectivity_Analysis(score):
    if score >  0:
      return 'Opinion'
    else:
      return 'FACT'

tweets['Analysis_Polarity'] = tweets['Polarity'].apply(get_Polarity_Analysis)

tweets['Analysis_Subjectivity'] = tweets['Subjectivity'].apply(get_Subjectivity_Analysis)

# Show the dataframe
tweets.head()


# In[46]:


tweets.to_csv("PolaritySubjectivityInnovation.csv")


# In[47]:


tweets.info()


# # SENTIMENT ANALYSIS

# In[48]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)


# In[49]:


plt.figure(figsize=(15,10)) 

# plt.style.use('seaborn-pastel')

plt.scatter(tweets['Polarity'], tweets['Subjectivity'], c=tweets['Polarity'], s=100, cmap='RdYlGn') 

plt.xlim(-1.1, 1.1)
plt.ylim(-0.1, 1.1) 
plt.title('Sentiment Analysis') 
plt.xlabel('Polarity') 
plt.ylabel('Subjectivity') 
plt.show(),


# In[50]:


# Show the value counts
tweets['Analysis_Polarity'].value_counts()


# In[51]:


# Plotting and visualizing the counts
plt.figure(figsize=(15,10)) 

plt.title('Polarity Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
tweets['Analysis_Polarity'].value_counts().plot(kind = 'bar')
plt.show()


# # Sentiment Analysis Pie Chart

# In[52]:


# Plotting and visualizing the counts
plt.figure(figsize=(15,10)) 

plt.title('Subjectivity Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
tweets['Analysis_Subjectivity'].value_counts().plot(kind = 'bar')
plt.show()


# In[53]:


# Show the value counts
tweets['Analysis_Subjectivity'].value_counts()


# # TOPIC MODELING

# In[54]:


get_ipython().system('pip install pyLDAvis')


# In[55]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()


from sklearn.feature_extrWFHTion.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[56]:


tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                max_df = 0.5, 
                                min_df = 10)
dtm_tf = tf_vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))
print(dtm_tf.shape)


# In[57]:


tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(tweets['cleaned_text'].values.astype('U'))
print(dtm_tfidf.shape)


# In[58]:


# for TF DTM
lda_tf = LatentDirichletAllocation(n_components =5, random_state=50)
lda_tf.fit(dtm_tf)
# for TFIDF DTM
lda_tfidf = LatentDirichletAllocation(n_components =5, random_state=50)
lda_tfidf.fit(dtm_tfidf)


# In[59]:


#for i,topic in enumerate(lda_tf.components_):
#print([tfidf_vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
#print('\n')


# In[60]:


for i,topic in enumerate(lda_tf.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tfidf_vectorizer.get_feature_names()[i] for i in topic.argsort()[-30:]])
    print('\n')


# In[61]:


pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)


# In[62]:


topic_values = lda_tf.transform(dtm_tf)
topic_values.shape


# In[63]:


tweets.head()


# In[64]:


tweets.columns


# In[65]:


tweets_1=tweets.replace({0:'Productivity',1:'Satisfaction',2:'Balance',3:'Family',4:'Commute'})


# In[66]:


tweets_1['Topic'] = topic_values.argmax(axis=1)


# In[67]:


tweets['Topic'] = topic_values.argmax(axis=1)


# In[68]:


tweets_1.head()


# In[69]:


tweets_1.columns


# In[70]:


import seaborn as sns


plt.figure(figsize=(40,25)) 

g=sns.lmplot(x="Polarity", y="Subjectivity", hue='Topic', data=tweets, fit_reg=False, legend=False,palette="GnBu_d", col='Topic', legend_out=True)
 
# # Move the legend to an empty part of the plot
# plt.legend(loc='lower right')
 
plt.show()


# In[71]:


tweets_2= tweets_1.groupby(['Topic'])['Analysis_Polarity'].value_counts().unstack('Topic').transpose()

tweets_2


# In[72]:


tweets_2.info()


# In[73]:


tweets_2['Total'] = tweets_2.sum(axis=1)


# In[74]:


tweets_2.columns


# In[75]:


for i in tweets_2:
    tweets_2[i] = round(tweets_2[i]*100/tweets_2.Total)

tweets_2


# In[76]:


tweets_2=tweets_2.fillna(0)


# In[77]:


#Conduct Polarity Topic Analysis using Tableau

tweets_2.to_excel("Polarity_Topic Modeling.xlsx")


# In[78]:


tweets_2= tweets_2.drop(['Total'], axis=1)


# In[79]:


tweets_2.head()


# In[80]:


#Plotting and visualizing the counts
plt.figure(figsize=(20,15)) 

topic = ['Productivity','Satisfaction','Balance','Family','Commute']
sentiment = ['Negative', 'Neutral', 'Positive']
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('%')
p1=plt.bar(topic,tweets_2['Negative'], color='#355E3B')
p2=plt.bar(topic,tweets_2['Neutral'], color='#00A36C', bottom=tweets_2['Negative'])
p3=plt.bar(topic,tweets_2['Positive'], color='#2AAA8A', bottom=tweets_2['Neutral']+tweets_2['Negative'])
plt.xticks(topic, rotation=90)
plt.xlabel("Topic")
plt.legend((p1[0], p2[0], p3[0]),('Negative', 'Neutral', 'Positive'),fontsize=12, ncol=4, framealpha=0, fancybox=True, loc='upper center')

plt.show()


# In[ ]:





#%%
