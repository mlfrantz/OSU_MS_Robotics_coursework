__author__ = "mlfrantz"

import praw
import pandas as pd
# import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reddit = praw.Reddit(client_id='3rxCdfotkQjLhQ',
                     client_secret='Dt_QBhUB13Sbh9zCs9G2ENeoeuU',
                     user_agent='MyResearchApp by /u/bigbeaver1',
                     username='bigbeaver1',
                     password='OrutomildE?'
                     )

# .hot(limit=1) # for hot items
# .top(time_filter='all', limit=1)

limit = 1
subreddit = reddit.subreddit('Ayahuasca').top(time_filter='all', limit=limit)
analyzer = SentimentIntensityAnalyzer()

# Pandas dataframe object will have the below columns
columns = ['ID', 'Title', 'Author', 'Votes', 'Number of Comments','Upvote Ratio',
 'Stickied', 'Flair', 'Distinguished', 'Permalink', 'URL']

comment_columns = ['Title', 'Author', 'Body', 'Num Replies', 'Votes',
 'ID', 'Link ID', 'Parent ID', 'Permalink', 'Negative', 'Neutral', 'Positive', 'Compound']

data = []
comment_data = []
for i, submission in enumerate(subreddit): # Object under subreddit is a submission object
    print(limit-i)
    #print("Title:", submission.title)
    # print("Author:", submission.author)
    # print("Votes:", submission.score)
    # print("Number of Comments:", submission.num_comments)
    # print("Upvote Ratio:", submission.upvote_ratio)
    # print("Stickied:", submission.stickied)
    # print("Link Flair Text:", submission.link_flair_text)
    # print("Distinuguished:", submission.distinguished)
    # print("URL:", submission.url)
    # sub_info = [submission.id, submission.title, submission.author, submission.score, submission.num_comments,
    #  submission.upvote_ratio, submission.stickied, submission.link_flair_text, submission.distinguished, submission.permalink, submission.url]
    # data.append(sub_info)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        # This for loop will go through all the coments in the comment tree
        # print("Comment Author:", comment.author)
        # print("Body:", comment.body)
        # print("Replies:", len(comment.replies))
        # print("Score:", comment.score)
        # print("Link ID:", comment.link_id)
        # print("Parent ID:", comment.parent_id)
        # print("Permalink:", comment.permalink)
        score = analyzer.polarity_scores(comment.body)
        comment_info = [submission.title, comment.author, comment.body, len(comment.replies), comment.score,
         comment.id, comment.link_id, comment.parent_id, comment.permalink, score['neg'], score['neu'], score['pos'], score['compound']]
        comment_data.append(comment_info)
        # print(analyzer.polarity_scores(comment.body))


# df = pd.DataFrame(data, columns=columns)
# df.to_csv('Ayahuasca_hot.csv')
# df = pd.DataFrame(comment_data, columns=comment_columns)
# df.to_csv('Ayahuasca_hot_100_comments.csv')
df = pd.DataFrame(comment_data, columns=comment_columns)
df.to_csv('Ayahuasca_comments_sentiment.csv')
