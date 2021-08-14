#importing packages

#Packages for Data manipulation/Graphing
from matplotlib import colors
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

#Bokeh and Plotly for interactive graphs
import plotly_express as px
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.palettes import Category20c
from bokeh.models import HoverTool

# Python File containing constants
import ModelResults

#Machine Learning Packages for Predicting
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### SAVED MODEL RESULTS 
DOJA_CAT_MODEL_RESULTS = ModelResults.DOJA_CAT_MODEL_RESULTS
TYLER_MODEL_RESULTS = ModelResults.TYLER_MODEL_RESULTS
MAROON_MODEL_RESULTS = ModelResults.MAROON_MODEL_RESULTS

### FOLDERS FOR READING IN IMAGES AND CHART DATA
IMAGES_FILE_PATH = "Album Images/"
DONUT_FILE_PATH = "Donut Data/"
MODEL_RESULTS_FILE_PATH = "Model Results Data/"
TWITTER_WORDCLOUD_FILE_PATH = "Twitter Cloud Images/"


### FUNCTION DEFINITIONS ###

# For comparative analysis construct numerical categories for the three types of sentiments
def get_label(polarity):
    if polarity > 0:
        return 1
    elif polarity == 0:
        return 0
    else:
        return -1


### START OF CODE - HEADER

# Sets the tab title on your browser and layout 
st.set_page_config(
    page_title="HypeLoop: Tweet Driven Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Header
subheader_text = """**Overview**\nHow does Twitter feel about Album releases, and how does it evolve overtime?, Hypeloop uses a selection of Naturual Language Processing Models to give an insight how an artists,
            fanbase feel about their album release, from before the release, the day of, and beyond. Choose an option from the sidebar to get started!"""

st.title("HypeLoop")
st.subheader(subheader_text)


# Side Bar Options

st.sidebar.title("Options")

album_option = st.sidebar.selectbox('Trending Albums',
                     ('Doja Cat: Planet Her',
                      "Tyler the Creator: Call Me If You Get Lost",
                      "Maroon 5: JORDI"))



# Code to set most of the data, image and other file paths for each album

col_left, col_image, col_right = st.beta_columns([3,6,3])

# DOJA CAT
if album_option == "Doja Cat: Planet Her":
    

    #Album Thumbnail and Title
    with col_image:
        st.markdown(f"<h1 style='text-align: center; color: #000000;'>{album_option}</h1>", unsafe_allow_html=True)
        st.image(IMAGES_FILE_PATH + "Doja Cat_ Planet Her.jpeg")


    hashtag_name = '#PlanetHer'
    TWEET_FILE_NAME = 'Doja.csv'

    #Release Date to print to dashboard
    RELEASE_DATE_STR = "June 25th, 2021"

    ## Hard coded Model Results Compiled
    MODEL_RESULTS_FILE_NAME = MODEL_RESULTS_FILE_PATH +  "DojaCatModelResults.csv"
    MODEL_RESULTS_STR = DOJA_CAT_MODEL_RESULTS


    #File for donut data
    DONUT_FILE_NAME = DONUT_FILE_PATH + "DojaCatDonutData.csv"

    #File for Word Cloud Image
    TWITTER_WORDCLOUD = TWITTER_WORDCLOUD_FILE_PATH + "DojaCatTwitterWordCloud.png"
    
    # Assigning Start/End for Bokeh Plot of Tweet Counts
    RELEASE_DATE_DRILL_DOWN_START = dt.datetime(2021,6,24)
    RELEASE_DATE_DRILL_DOWN_END = dt.datetime(2021,6,26)



# TYLER THE CREATOR
if album_option == "Tyler the Creator: Call Me If You Get Lost":
    
    #Album Thumbnail and Title
    with col_image:
        st.markdown(f"<h1 style='text-align: center; color: #000000;'>{album_option}</h1>", unsafe_allow_html=True)
        st.image(IMAGES_FILE_PATH + "Tyler-The-Creator-Call-Me-If-You-Get-Lost.png")

    # Setting Files and String Variables
    hashtag_name = '#CallMeIfYouGetLost'
    TWEET_FILE_NAME = 'Tyler.csv'
    MODEL_RESULTS_FILE_NAME = MODEL_RESULTS_FILE_PATH + "TylerModelResults.csv"
    DONUT_FILE_NAME = DONUT_FILE_PATH + "TylerDonutData.csv"
    TWITTER_WORDCLOUD = TWITTER_WORDCLOUD_FILE_PATH + 'TylerTwitterWordCloud.png'
    RELEASE_DATE_STR = "June 25th, 2021"

    
    MODEL_RESULTS_STR = TYLER_MODEL_RESULTS

    RELEASE_DATE_DRILL_DOWN_START = dt.datetime(2021,6,24)
    RELEASE_DATE_DRILL_DOWN_END = dt.datetime(2021,6,26)

#MAROON 5
if album_option == "Maroon 5: JORDI":

    #Album Thumbnail and Title
    with col_image:
        st.markdown(f"<h1 style='text-align: center; color: #000000;'>{album_option}</h1>", unsafe_allow_html=True)
        st.image(IMAGES_FILE_PATH + "Maroon5.jpg", use_column_width = 'auto')

    # Setting Files and String Variables
    hashtag_name = '#JORDI'
    TWEET_FILE_NAME = 'Maroon.csv'

    RELEASE_DATE_STR = "June 11th, 2021"
    DONUT_FILE_NAME = DONUT_FILE_PATH + "MaroonDonutData.csv"
    TWITTER_WORDCLOUD = TWITTER_WORDCLOUD_FILE_PATH + 'JordiTwitterWordCloud.png'
    MODEL_RESULTS_STR = MAROON_MODEL_RESULTS
    MODEL_RESULTS_FILE_NAME = MODEL_RESULTS_FILE_PATH + "MaroonModelResults.csv"

    RELEASE_DATE_DRILL_DOWN_START = dt.datetime(2021,6,10)
    RELEASE_DATE_DRILL_DOWN_END = dt.datetime(2021,6,13)


# Reading in Data, this read is done each time a new album option is selected

df = pd.read_csv(TWEET_FILE_NAME)

#converting to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = pd.to_datetime(df['date'])

#stripping the minutes and seconds of the date
df['created_at'] = df['created_at'] - pd.to_timedelta(df['created_at'].dt.second, unit = 's')
df['created_at'] = df['created_at'] - pd.to_timedelta(df['created_at'].dt.minute, unit = 'm')


### Splitting Dataset into Training and Testing
# train is 75% and test is 25%
model_train = df[:int(len(df) * .75)]
model_test = df[int(len(df) * .75):]

# Converting text to a matrix of token counts
vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(2,2),min_df=8)
tf_features_train = vectorizer.fit_transform(model_train['text'])
tf_features_test = vectorizer.transform(model_test['text'])

# Assigning Sentiment based on polarity
train_labels = model_train['polarity'].apply(get_label)
test_labels = model_test['polarity'].apply(get_label)


#####################################
#### START OF DATA VISUALIZATION ####
#####################################


# Columns showing total tweets, unique users,and the release date
st.markdown("<hr/>", unsafe_allow_html=True)

column_total_tweets, column_unique_users, column_release_date = st.beta_columns(3)

# Total Tweets
with column_total_tweets:
    st.markdown("**Total Tweets**")
    number = len(df)
    st.markdown(f"<h1 style='text-align: center; color: #000000;'>{number}</h1>", unsafe_allow_html=True)

# Total Unique Users
with column_unique_users:
    st.markdown("**Unique Users**")
    number = len(df['user_id_str'].unique())
    st.markdown(f"<h1 style='text-align: center; color: #000000;'>{number}</h1>", unsafe_allow_html=True)

# Release Date
with column_release_date:
        st.subheader("**Release Date**")
        
        st.markdown(f"<h1 style='text-align: center; color: #000000;'>{RELEASE_DATE_STR}</h1>", unsafe_allow_html=True)

# Horizontal Rule
st.markdown("<hr/>", unsafe_allow_html=True)

# Users who made the most tweets with the target hashtag
column_total_tweets, column_top_fans, column_release_date = st.beta_columns([2,5,2])
with column_top_fans:
    top_fans_str = "Top 10 Tweeters using target hashtag: " + hashtag_name
    st.markdown(f"<h2 style='text-align: center; color: #000000;'>{top_fans_str}</h2>", unsafe_allow_html=True)
    df_top_10_users = pd.DataFrame(df['username'].value_counts().head(10))

    df_top_10_users.columns = ['Number Of Tweets Made Containing the Hashtag ' + hashtag_name]

    st.write(df_top_10_users)

st.markdown("<hr/>", unsafe_allow_html=True)

# Correlation Matrix of Tweet Post Metrics

column_total_tweets, column_corr, column_release_date = st.beta_columns([2,8,2])

with column_corr:
    fig, ax = plt.subplots(figsize = (12,5))

    df_heatmap = pd.DataFrame(df[['likecount','replycount','retweetcount', 'hashtagsCount', 'textCount']])
    df_heatmap.columns = ['Likes', 'Replies', 'Retweets', 'Number Hashtags', 'Number Words']

    ax = sns.heatmap(df_heatmap.corr(), annot=True, cmap = "magma")
    ax.set_title("Tweet Correlation Matrix", size = 28)

    st.pyplot(fig)

# Associated Hashtags donut Chart
margin_col_left, col_donut_chart, margin_col_right = st.beta_columns([5,6,5])

with col_donut_chart:
    st.header("Highest Occuring Hashtags")

donut_data = pd.read_csv(DONUT_FILE_NAME)

p_donut = figure(plot_height=1000, plot_width = 1400, title="", toolbar_location=None,
           tools="hover", tooltips="@hashtag_names: @value_100{0.2f} %", x_range=(-.5, .5))

p_donut.annular_wedge(x=0, y=1,  inner_radius=0.15, outer_radius=0.25, direction="anticlock",
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color',  source=donut_data, legend_group = 'hashtag_names')

p_donut.axis.axis_label=None
p_donut.axis.visible=False
p_donut.grid.grid_line_color = None


st.bokeh_chart(p_donut)


## Twitter Logo Word Cloud
margin_col_left, col_word_cloud, margin_col_right = st.beta_columns([5,6,5])

# Display the generated image:
with col_word_cloud:
    wordcloud_header = "Tweet Word Frequency"
    st.markdown(f"<h2 style='text-align: center; color: #000000;font-family: Sans-Serif;'>{wordcloud_header}</h2>", unsafe_allow_html=True)
    st.image(TWITTER_WORDCLOUD)


### Line plot of sentiment counts by Day
p_line_sentiment = figure(
    title = "Sentiment of Tweets containing " + hashtag_name,
    x_axis_type="datetime",
    x_axis_label='Day',
    y_axis_label='Number of Associated Sentiments'
)


# Plot Each Sentiment
p_line_sentiment.line(x = df[df['sentiment'] == 'Neutral']['date'].value_counts().sort_index(ascending=True).index,
                    y = df[df['sentiment'] == 'Neutral']['date'].value_counts().sort_index(ascending=True).values,
                     line_color = 'green',
                     legend_label = 'Neutral')

p_line_sentiment.line(x = df[df['sentiment'] == 'Positive']['date'].value_counts().sort_index(ascending=True).index,
                    y = df[df['sentiment'] == 'Positive']['date'].value_counts().sort_index(ascending=True).values,
                     line_color = 'blue',
                     legend_label = 'Positive')

p_line_sentiment.line(x = df[df['sentiment'] == 'Negative']['date'].value_counts().sort_index(ascending=True).index,
                    y = df[df['sentiment'] == 'Negative']['date'].value_counts().sort_index(ascending=True).values,
                     line_color = 'red',
                     legend_label = 'Negative')


p_line_sentiment.legend.location = "top_left"
p_line_sentiment.legend.click_policy="hide"

st.bokeh_chart(p_line_sentiment, use_container_width=True)

# Drill down to hour
drill_down_option = st.selectbox('Select How the Hourly Tweet Count Will be Displayed',
                    ('Total Tweets', 'Count by Sentiment'))

st.subheader("Count of Tweets On The Day Of the Album's Release")

df_by_hour = df.copy()

#Grabbing days of interest
df_by_hour = df_by_hour[df_by_hour['created_at'] < RELEASE_DATE_DRILL_DOWN_END]
df_by_hour = df_by_hour[df_by_hour['created_at'] > RELEASE_DATE_DRILL_DOWN_START]

# Plot for total tweets
df_date_release = pd.DataFrame(df_by_hour['created_at'].value_counts())
df_date_release.columns = ['Tweet Count']
df_date_release = df_date_release.sort_index(ascending=True)


#Creating DF for each sentiment
df_date_release_negative = pd.DataFrame(df_by_hour[df_by_hour['sentiment'] == 'Negative']['created_at'].value_counts())
df_date_release_negative.columns = ['Tweet Count']
df_date_release_negative['sentiment'] = 'Negative'
df_date_release_negative = df_date_release_negative.sort_index(ascending=True)

df_date_release_neutral = pd.DataFrame(df_by_hour[df_by_hour['sentiment'] == 'Neutral']['created_at'].value_counts())
df_date_release_neutral.columns = ['Tweet Count']
df_date_release_neutral['sentiment'] = 'Neutral'
df_date_release_neutral = df_date_release_neutral.sort_index(ascending=True)

df_date_release_positive = pd.DataFrame(df_by_hour[df_by_hour['sentiment'] == 'Positive']['created_at'].value_counts())
df_date_release_positive.columns = ['Tweet Count']
df_date_release_positive['sentiment'] = 'Positive'
df_date_release_positive = df_date_release_positive.sort_index(ascending=True)


# Two Drill down options: Either display all the Tweets or break down the counts by sentiment

if drill_down_option == 'Total Tweets':

    p_line_hour = figure(
    x_axis_type="datetime",
    x_axis_label='Hour',
    y_axis_label='Count of Tweets',
    tools = 'hover,pan,wheel_zoom,box_zoom,reset'
    )   

    hover = p_line_hour.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Count","$y{0}")
    ]
    

    p_line_hour.line(x = df_date_release.index,
                        y = df_date_release['Tweet Count'])

    

    st.bokeh_chart(p_line_hour, use_container_width=True)


# Count by Sentiment
if drill_down_option == 'Count by Sentiment':
    p_line_sentiment_hour = figure(
    x_axis_type="datetime",
    x_axis_label='Hour',
    y_axis_label='Number of Associated Sentiments',
    tools = 'hover,pan,wheel_zoom,box_zoom,reset'
    ) 

    hover = p_line_sentiment_hour.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Count","$y{0}")
    ]

    p_line_sentiment_hour.line(x = df_date_release_neutral.index,
                        y = df_date_release_neutral['Tweet Count'],
                        line_color = "blue",
                        legend_label = "Neutral")

    p_line_sentiment_hour.line(x = df_date_release_positive.index,
                        y = df_date_release_positive['Tweet Count'],
                        line_color = "green",
                        legend_label = "Positive")

    p_line_sentiment_hour.line(x = df_date_release_negative.index,
                        y = df_date_release_negative['Tweet Count'],
                        line_color = "red",
                        legend_label = "Negative")

    st.bokeh_chart(p_line_sentiment_hour, use_container_width=True)


## Sentiment Distribution Scatter Plot
pol = df['polarity']
sub = df['subjectivity']
sen = df['sentiment']

#columns to make the visualization smaller and have margins
column_margin_left, column_scatter_sentiment, column_margin_right = st.beta_columns([1,8,2])

with column_scatter_sentiment:
    st.header("Distribution of Sentiment")
    fig, ax = plt.subplots(figsize=(12,6))
    ax = sns.scatterplot(x = pol, y = sub, data = df, hue = "sentiment", palette = "magma")

    ax.set_xlabel("Polarity", size = 16)
    ax.set_ylabel("Subjectivity", size = 16)

    st.pyplot(fig)


## Correlation Matrix
contingency_table = pd.crosstab(df['hype_loop'], df['sentiment'])
column_margin_left, column_sentiment_corr, column_margin_right = st.beta_columns([1,8,2])

with column_scatter_sentiment:
    # reporting as a percentage
    st.header("Twitter Sentiment Based on Time Posted")
    contingency_pct = pd.crosstab(df['sentiment'], df['hype_loop'], normalize='index')

    fig = plt.figure(figsize = (12,6))
    sns.heatmap(contingency_pct, annot = True, cmap = "magma")
    plt.xlabel("Time Posted", fontsize = 16)
    plt.ylabel("Sentiment", fontsize = 16)
    st.pyplot(fig)

#Options for each model

SUMMARY_STR = "Model Metric Summary"


st.markdown(f"<h1 style='text-align: center; color: #000000;'>{SUMMARY_STR}</h1>", unsafe_allow_html=True)


option_model = st.selectbox('Select The Model',
                    ('Logistic Regression','Support Vector Machine',
                      "Multinomial Naive Bayes","Comparative Analysis and Regression"))



#Columns and margins for the model output
column_margin_left, column_results, column_margin_right = st.beta_columns([5,6,5])
column_margin_left, column_results_corr, column_margin_right = st.beta_columns([2.5,5,3])

column_margin_left, column_results_1, column_margin_right = st.beta_columns([5,6,5])
column_margin_left, column_results_corr_1, column_margin_right = st.beta_columns([2.5,5,3])

column_margin_left, column_results_2, column_margin_right = st.beta_columns([5,6,5])
column_margin_left, column_results_corr_2, column_margin_right = st.beta_columns([2.5,5,3])



#Logistic Regression
if option_model == 'Logistic Regression':

    #UNI
    clf = sklearn.linear_model.LogisticRegression(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report =  sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])

    # Confusion Matrix
    with column_results:
        st.text("-" + class_report)
        with column_results_corr:
        #Confusion Matrix
            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("Logisitic Model Prediction (Uni)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

    #UNI + BI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = sklearn.linear_model.LogisticRegression(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])


    with column_results_1:
        st.text("-" + class_report)
        with column_results_corr_1:
            fig, ax = plt.subplots(figsize = (10,6))

            # Confusion Matrix
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("Logisitic Model Prediction (Uni/Bi)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

    #Uni + Bi + Tri
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,3))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = sklearn.linear_model.LogisticRegression(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])

    with column_results_2:
        
        st.text("-" + class_report)
        with column_results_corr_2:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("Logisitic Model Prediction (Uni/Bi/Tri)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)
    

# Support Vector Machine

if option_model == 'Support Vector Machine':
    #UNI

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,1))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = sklearn.svm.LinearSVC(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])

    with column_results:
        st.text("-" + class_report)
        with column_results_corr:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("SVM Model Prediction (Uni)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

    #UNI + BI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = sklearn.svm.LinearSVC(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])

    with column_results_1:
        st.text("-" + class_report)
        with column_results_corr_1:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("SVM Model Prediction (Uni/Bi)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

    #UNI + BI + TRI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,3))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = sklearn.svm.LinearSVC(max_iter=9000)
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])
    with column_results_2:
        st.text("-" + class_report)
        with column_results_corr_2:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("SVM Model Prediction (Uni/Bi/Tri)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

# Multinomial Naive Bayes

if option_model == "Multinomial Naive Bayes":
    #UNI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,1))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])

    with column_results:
        st.text("-" + class_report)
        with column_results_corr:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("MNB Model Prediction (Uni)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)
    
    # UNI + BI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])


    with column_results_1:
        st.text("-" + class_report)  
        with column_results_corr_1:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("MNB Model Prediction (Uni/Bi)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

    # UNI + BI + TRI
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(model_train['text'])
    tf_features_test = vectorizer.transform(model_test['text'])
    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)
    predictions = clf.predict(tf_features_test)

    class_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])


    with column_results_2:
        st.text("-" + class_report)  
        with column_results_corr_2:

            fig, ax = plt.subplots(figsize = (10,6))
            cm = confusion_matrix(test_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap = "magma")
            plt.title("MNB Model Prediction (Uni/Bi/Tri)", fontsize = 28)
            plt.xlabel("Predicted Model Class", fontsize = 16)
            plt.ylabel("Test Dataset", fontsize = 16)
            st.pyplot(fig)

        

if option_model == "Comparative Analysis and Regression":
    
    df_model_results = pd.read_csv(MODEL_RESULTS_FILE_NAME)
    df_model_results = df_model_results.drop(columns = "Unnamed: 0").sort_index(ascending=True)

    margin_col_left, col_results_table, margin_col_right = st.beta_columns([3,8,3])

    with col_results_table:
        st.subheader("Comparing Results of Each Model")
        st.write(df_model_results)
        st.subheader("OLS Regression Results")
        st.text(MODEL_RESULTS_STR)

