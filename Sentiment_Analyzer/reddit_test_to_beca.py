__author__ = "mlfrantz"

import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reddit = praw.Reddit(client_id='3rxCdfotkQjLhQ',
                     client_secret='Dt_QBhUB13Sbh9zCs9G2ENeoeuU',
                     user_agent='MyResearchApp by /u/username',
                     username='username',
                     password='password'
                     )

# .hot(limit=1) # for hot items
# .top(time_filter='all', limit=1)

limit = 10
subreddit = reddit.subreddit('Ayahuasca').top(time_filter='all', limit=limit)
analyzer = SentimentIntensityAnalyzer()

# Pandas dataframe object will have the below columns
columns = ['ID', 'Title', 'Author', 'Votes', 'Number of Comments','Upvote Ratio',
 'Stickied', 'Flair', 'Distinguished', 'Permalink', 'URL']

comment_columns = ['Title', 'Author', 'Body', 'Num Replies', 'Votes',
 'ID', 'Link ID', 'Parent ID', 'Permalink', 'Negative', 'Neutral', 'Positive', 'Compound']

data = []
for i, submission in enumerate(subreddit): # Object under subreddit is a submission object
    sub_info = [submission.id, submission.title, submission.author, submission.score, submission.num_comments,
     submission.upvote_ratio, submission.stickied, submission.link_flair_text, submission.distinguished, submission.permalink, submission.url]
    data.append(sub_info)
    submission.comments.replace_more(limit=None)
df = pd.DataFrame(data, columns=columns)
df.to_csv('Ayahuasca_top.csv')

comment_data = []
for i, submission in enumerate(subreddit): # Object under subreddit is a submission object
    for comment in submission.comments.list():
        score = analyzer.polarity_scores(comment.body)
        comment_info = [submission.title, comment.author, comment.body, len(comment.replies), comment.score,
         comment.id, comment.link_id, comment.parent_id, comment.permalink, score['neg'], score['neu'], score['pos'], score['compound']]
        comment_data.append(comment_info)

df = pd.DataFrame(comment_data, columns=comment_columns)
df.to_csv('Ayahuasca_comments_sentiment.csv')
