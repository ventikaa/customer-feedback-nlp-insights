import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect, DetectorFactory, LangDetectException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models
import gensim
import gensim.corpora as corpora
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from nrclex import NRCLex
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import random

st.set_page_config(
    page_title="Advanced Customer Feedback NLP Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Advanced Customer Feedback NLP Insights")
st.markdown("""
Extract actionable insights from product reviews to guide product improvements.
This enhanced dashboard includes emotion classification, aspect-based sentiment analysis, review summarization, and more.
""")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", [
    "Data Overview", 
    "Sentiment Analysis", 
    "Emotion Classification",
    "Aspect-Based Sentiment Analysis",
    "Topic Modeling", 
    "Trend Analysis",
    "Review Summarization",
    "Actionable Insights"
])

DetectorFactory.seed = 0

@st.cache_data
def load_data():
    # For demonstration, we'll create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    n_reviews = 1000
    
    # Sample review texts with different sentiments, emotions, and aspects
    positive_reviews = [
        "Love this app! Very user friendly and intuitive interface.",
        "Great functionality, does exactly what I need it to do.",
        "The latest update fixed all my issues, fantastic work!",
        "Customer support was extremely helpful and responsive.",
        "This has improved my workflow significantly. Highly recommend!",
        "The design is sleek and modern. Very pleasant to use daily.",
        "Fast performance even with large files. Impressive!",
        "The onboarding process was smooth and easy to follow.",
        "This product has all the features I've been looking for.",
        "Reliable and stable. No crashes or bugs encountered."
    ]
    
    negative_reviews = [
        "The app crashes frequently, very frustrating to use.",
        "User interface is confusing and not intuitive at all.",
        "Customer support takes days to respond to simple queries.",
        "The latest update made everything slower. Very disappointed.",
        "Too expensive for the limited features it offers.",
        "The onboarding process is complicated and poorly explained.",
        "Sync issues between devices. Lost important data.",
        "Bugs everywhere. Doesn't work as advertised.",
        "Poor performance with large files. Crashes constantly.",
        "Lacks basic features that competitors have had for years."
    ]
    
    neutral_reviews = [
        "The app is okay, does what it needs to but nothing special.",
        "It's a decent product but there's room for improvement.",
        "I use it occasionally. It serves its purpose.",
        "Not the best, not the worst. Middle of the road product.",
        "Does the job but the interface could be more modern.",
        "Adequate for basic tasks but lacks advanced features.",
        "It's fine. I haven't had major issues but I'm not excited about it.",
        "Average performance. Gets the job done but slowly.",
        "The design is functional but not particularly appealing.",
        "I'll keep using it until I find something better."
    ]
    
    # Multilingual reviews
    multilingual_reviews = [
        "Excelente aplicación, muy fácil de usar. ¡Me encanta!",
        "Très déçu par ce produit. Il ne fonctionne pas comme annoncé.",
        "Das Update hat alles schlimmer gemacht. Sehr enttäuschend.",
        "このアプリは素晴らしいです。とても使いやすいです。",
        "应用程序经常崩溃，非常令人沮丧。"
    ]
    
    all_reviews = positive_reviews + negative_reviews + neutral_reviews + multilingual_reviews
    
    data = {
        'date': np.random.choice(dates, n_reviews),
        'review_text': np.random.choice(all_reviews, n_reviews),
        'rating': np.random.randint(1, 6, n_reviews),
        'product': np.random.choice(['App A', 'App B', 'App C'], n_reviews),
        'source': np.random.choice(['Google Play', 'App Store', 'Trustpilot'], n_reviews),
        'version': np.random.choice(['1.2.3', '1.3.0', '1.3.1', '1.4.0'], n_reviews)
    }
    
    df = pd.DataFrame(data)
    
    # Add some more varied text with specific aspects
    aspects = ['performance', 'price', 'design', 'usability', 'features', 'support', 'reliability']
    aspect_phrases = {
        'performance': ["runs slowly", "fast performance", "laggy interface", "quick response"],
        'price': ["too expensive", "good value", "pricey but worth it", "not worth the cost"],
        'design': ["sleek design", "ugly interface", "modern look", "outdated appearance"],
        'usability': ["easy to use", "complicated workflow", "intuitive interface", "steep learning curve"],
        'features': ["missing features", "all the features I need", "limited functionality", "comprehensive toolset"],
        'support': ["helpful support", "slow response", "knowledgeable team", "unhelpful agents"],
        'reliability': ["constant crashes", "rock solid", "unstable application", "dependable performance"]
    }
    
    for i in range(len(df)):
        if np.random.random() > 0.5:
            aspect = np.random.choice(aspects)
            phrase = np.random.choice(aspect_phrases[aspect])
            df.iloc[i, 1] = df.iloc[i, 1] + " " + phrase
    
    # Add some emotion-rich reviews
    emotion_reviews = [
        "I'm absolutely thrilled with this product! It's changed my life!",
        "So frustrated with the constant bugs and crashes. I'm furious!",
        "This app makes me so happy. It's exactly what I needed.",
        "I'm afraid to use this app after it deleted my important files.",
        "The surprise update with new features was a wonderful shock!",
        "It's so sad that such a promising app has so many issues."
    ]
    
    for i in range(50):  # Add emotion reviews to random rows
        idx = np.random.randint(0, len(df))
        df.iloc[idx, 1] = np.random.choice(emotion_reviews)
    
    # Create some anomaly dates with spikes in negative reviews
    anomaly_dates = [datetime(2023, 3, 15), datetime(2023, 5, 1)]
    for anomaly_date in anomaly_dates:
        anomaly_mask = (pd.to_datetime(df['date']).dt.date == anomaly_date.date())
        df.loc[anomaly_mask, 'review_text'] = np.random.choice(negative_reviews, size=anomaly_mask.sum())
        df.loc[anomaly_mask, 'rating'] = np.random.choice([1, 2], size=anomaly_mask.sum())
    
    return df

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
        return 'Positive', sentiment_scores['compound']
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative', sentiment_scores['compound']
    else:
        return 'Neutral', sentiment_scores['compound']
def analyze_emotion(text):
    try:
        emotion = NRCLex(text)
        return emotion.affect_dict
    except:
        return {}
def get_top_emotions(emotion_dict, top_n=3):
    if not emotion_dict:
        return {}
    
    emotion_counts = {}
    for emotions in emotion_dict.values():
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Get top N emotions
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_emotions[:top_n])

# Aspect-based sentiment analysis (simulated for demo)
def analyze_aspect_sentiment(text):
    aspects = ['performance', 'price', 'design', 'usability', 'features', 'support', 'reliability']
    results = {}
    
    for aspect in aspects:
        if aspect in text.lower():
            # Simulate sentiment for each aspect
            sentiment_score = random.uniform(-1, 1)
            if sentiment_score >= 0.2:
                sentiment = 'Positive'
            elif sentiment_score <= -0.2:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            results[aspect] = {
                'sentiment': sentiment,
                'score': sentiment_score
            }
    
    return results

# Language detection function
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

# Translate text function
def translate_text(text, dest='en'):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=dest)
        return translation.text
    except:
        return text

# Text summarization function
def summarize_text(text, sentences_count=2):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return ' '.join([str(sentence) for sentence in summary])
    except:
        return "Unable to generate summary for this text."

# Load the data
df = load_data()

# Preprocess the text
df['cleaned_text'] = df['review_text'].apply(preprocess_text)

# Perform sentiment analysis
sentiment_results = df['review_text'].apply(analyze_sentiment)
df['sentiment'] = sentiment_results.apply(lambda x: x[0])
df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])

# Perform emotion analysis
df['emotion_dict'] = df['review_text'].apply(analyze_emotion)
df['top_emotions'] = df['emotion_dict'].apply(get_top_emotions)

# Perform aspect-based sentiment analysis
df['aspect_sentiment'] = df['review_text'].apply(analyze_aspect_sentiment)

# Detect language
df['language'] = df['review_text'].apply(detect_language)

# Data Overview Section
if options == "Data Overview":
    st.header("📁 Data Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Dataset Info")
        st.write(f"Total Reviews: {len(df)}")
        st.write(f"Time Period: {df['date'].min().date()} to {df['date'].max().date()}")
        
        st.subheader("Data Sources")
        source_counts = df['source'].value_counts()
        fig = px.pie(values=source_counts.values, names=source_counts.index, title="Review Sources")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Rating Distribution")
    rating_counts = df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values, 
                 labels={'x': 'Rating', 'y': 'Count'}, 
                 title="Distribution of Ratings")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Review Length Analysis")
    df['review_length'] = df['review_text'].apply(lambda x: len(x.split()))
    fig = px.histogram(df, x='review_length', nbins=20, title="Distribution of Review Lengths")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Language Distribution")
    language_counts = df['language'].value_counts()
    fig = px.pie(values=language_counts.values, names=language_counts.index, title="Review Languages")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis Section
elif options == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                     title="Overall Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment by Rating")
        sentiment_by_rating = pd.crosstab(df['rating'], df['sentiment'])
        fig = px.bar(sentiment_by_rating, barmode='group', title="Sentiment Distribution by Rating")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Word Clouds by Sentiment")
    
    sentiment_option = st.selectbox("Select Sentiment", ['Positive', 'Negative', 'Neutral'])
    
    # Generate word cloud
    text = ' '.join(df[df['sentiment'] == sentiment_option]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for {sentiment_option} Reviews')
    st.pyplot(fig)
    
    st.subheader("Detailed Sentiment Analysis")
    
    # Show positive and negative reviews
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Positive Reviews**")
        positive_reviews = df[df['sentiment'] == 'Positive'].nlargest(5, 'sentiment_score')
        for i, row in positive_reviews.iterrows():
            st.write(f"Rating: {row['rating']} - {row['review_text']}")
            st.write("---")
    
    with col2:
        st.write("**Top Negative Reviews**")
        negative_reviews = df[df['sentiment'] == 'Negative'].nsmallest(5, 'sentiment_score')
        for i, row in negative_reviews.iterrows():
            st.write(f"Rating: {row['rating']} - {row['review_text']}")
            st.write("---")

# Emotion Classification Section
elif options == "Emotion Classification":
    st.header("😢😄😠 Emotion Classification")
    
    # Extract all emotions and their counts
    all_emotions = {}
    for emotions in df['top_emotions']:
        for emotion, count in emotions.items():
            all_emotions[emotion] = all_emotions.get(emotion, 0) + count
    
    # Create emotion distribution chart
    if all_emotions:
        st.subheader("Emotion Distribution")
        emotion_df = pd.DataFrame(list(all_emotions.items()), columns=['Emotion', 'Count'])
        fig = px.bar(emotion_df, x='Emotion', y='Count', title='Distribution of Emotions in Reviews')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create radar chart for emotions by product
        st.subheader("Emotion Profile by Product")
        
        # Prepare data for radar chart
        products = df['product'].unique()
        emotions_list = list(all_emotions.keys())
        
        # Create a DataFrame for radar chart
        radar_data = []
        for product in products:
            product_emotions = {}
            product_reviews = df[df['product'] == product]
            
            for emotions in product_reviews['top_emotions']:
                for emotion, count in emotions.items():
                    product_emotions[emotion] = product_emotions.get(emotion, 0) + count
            
            # Normalize by number of reviews
            for emotion in emotions_list:
                product_emotions[emotion] = product_emotions.get(emotion, 0) / len(product_reviews)
            
            radar_data.append(product_emotions)
        
        radar_df = pd.DataFrame(radar_data, index=products)
        radar_df = radar_df.fillna(0)
        
        # Create radar chart
        fig = go.Figure()
        
        for product in products:
            fig.add_trace(go.Scatterpolar(
                r=radar_df.loc[product].values,
                theta=emotions_list,
                fill='toself',
                name=product
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(radar_df.max())]
                )),
            showlegend=True,
            title="Emotion Profile by Product"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show reviews for selected emotion
        st.subheader("Explore Reviews by Emotion")
        selected_emotion = st.selectbox("Select Emotion", list(all_emotions.keys()))
        
        # Find reviews with the selected emotion
        emotion_reviews = []
        for idx, row in df.iterrows():
            if selected_emotion in row['top_emotions']:
                emotion_reviews.append((row['review_text'], row['rating'], row['top_emotions'][selected_emotion]))
        
        # Sort by emotion strength
        emotion_reviews.sort(key=lambda x: x[2], reverse=True)
        
        # Display top reviews
        st.write(f"**Top reviews with '{selected_emotion}' emotion:**")
        for i, (review, rating, strength) in enumerate(emotion_reviews[:5]):
            st.write(f"**Rating: {rating}** (Emotion strength: {strength})")
            st.write(review)
            st.write("---")
    else:
        st.warning("No emotion data available. Please check your data or emotion analysis setup.")

# Aspect-Based Sentiment Analysis Section
elif options == "Aspect-Based Sentiment Analysis":
    st.header("🔍 Aspect-Based Sentiment Analysis")
    
    # Extract all aspects and their sentiment
    aspect_data = []
    for idx, row in df.iterrows():
        for aspect, sentiment_info in row['aspect_sentiment'].items():
            aspect_data.append({
                'aspect': aspect,
                'sentiment': sentiment_info['sentiment'],
                'score': sentiment_info['score'],
                'product': row['product'],
                'source': row['source']
            })
    
    if aspect_data:
        aspect_df = pd.DataFrame(aspect_data)
        
        # Create aspect sentiment heatmap
        st.subheader("Aspect Sentiment Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        
        # Normalize by row to show percentage
        heatmap_data_percent = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data_percent, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax)
        ax.set_title('Aspect Sentiment Distribution')
        st.pyplot(fig)
        
        # Show aspect sentiment by product
        st.subheader("Aspect Sentiment by Product")
        
        product_aspect = aspect_df.groupby(['product', 'aspect', 'sentiment']).size().unstack(fill_value=0)
        product_aspect_percent = product_aspect.div(product_aspect.sum(axis=1), axis=0)
        
        # Create subplots for each product
        products = aspect_df['product'].unique()
        fig, axes = plt.subplots(1, len(products), figsize=(18, 6))
        
        if len(products) == 1:
            axes = [axes]
        
        for i, product in enumerate(products):
            product_data = product_aspect_percent.loc[product]
            sns.heatmap(product_data, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[i])
            axes[i].set_title(f'Aspect Sentiment for {product}')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show detailed aspect analysis
        st.subheader("Detailed Aspect Analysis")
        
        selected_aspect = st.selectbox("Select Aspect", aspect_df['aspect'].unique())
        
        aspect_reviews = []
        for idx, row in df.iterrows():
            if selected_aspect in row['aspect_sentiment']:
                sentiment_info = row['aspect_sentiment'][selected_aspect]
                aspect_reviews.append({
                    'review': row['review_text'],
                    'rating': row['rating'],
                    'sentiment': sentiment_info['sentiment'],
                    'score': sentiment_info['score']
                })
        
        # Sort by sentiment score
        aspect_reviews.sort(key=lambda x: x['score'])
        
        # Display negative and positive reviews for the aspect
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Negative reviews about {selected_aspect}:**")
            negative_aspect_reviews = [r for r in aspect_reviews if r['sentiment'] == 'Negative']
            for i, review in enumerate(negative_aspect_reviews[:3]):
                st.write(f"Rating: {review['rating']}")
                st.write(review['review'])
                st.write("---")
        
        with col2:
            st.write(f"**Positive reviews about {selected_aspect}:**")
            positive_aspect_reviews = [r for r in aspect_reviews if r['sentiment'] == 'Positive']
            for i, review in enumerate(positive_aspect_reviews[:3]):
                st.write(f"Rating: {review['rating']}")
                st.write(review['review'])
                st.write("---")
    else:
        st.warning("No aspect-based sentiment data available. Please check your data or aspect analysis setup.")

# Topic Modeling Section
elif options == "Topic Modeling":
    st.header("🔍 Topic Modeling")
    
    st.info("""
    Topic modeling helps identify recurring themes in customer feedback. 
    This can reveal what aspects of your product users are talking about most.
    """)
    
    # Prepare data for LDA
    reviews = df['cleaned_text'].tolist()
    
    # Create dictionary and corpus
    tokens = [review.split() for review in reviews]
    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(token) for token in tokens]
    
    # Build LDA model
    num_topics = st.slider("Number of Topics", min_value=3, max_value=8, value=5)
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Display topics
    st.subheader(f"Discovered Topics (Top 10 Words)")
    
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        st.write(f"**Topic {topic[0]}**: {topic[1]}")
    
    # Visualize topics
    st.subheader("Topic Visualization")
    
    # Prepare the visualization
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    
    # Since we can't directly display pyLDAvis in Streamlit, we'll create an alternative visualization
    topic_words = {}
    for topic_id in range(num_topics):
        topic_words[topic_id] = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
    
    # Create a heatmap of topic-word distribution
    words = list(set([word for topic in topic_words.values() for word in topic]))
    heatmap_data = np.zeros((len(words), num_topics))
    
    for topic_id, words_list in topic_words.items():
        for word in words_list:
            if word in words:
                heatmap_data[words.index(word), topic_id] = 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=[f"Topic {i}" for i in range(num_topics)], 
                yticklabels=words, cmap="Blues", ax=ax)
    plt.title("Topic-Word Distribution")
    st.pyplot(fig)
    
    # Assign topics to documents
    df['topic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(df))]
    
    st.subheader("Topic Distribution")
    topic_counts = df['topic'].value_counts().sort_index()
    fig = px.bar(x=[f"Topic {i}" for i in topic_counts.index], y=topic_counts.values,
                 labels={'x': 'Topic', 'y': 'Number of Reviews'},
                 title="Distribution of Reviews Across Topics")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sentiment by Topic")
    topic_sentiment = pd.crosstab(df['topic'], df['sentiment'])
    fig = px.bar(topic_sentiment, barmode='group', 
                 title="Sentiment Distribution Across Topics",
                 labels={'value': 'Count', 'topic': 'Topic', 'variable': 'Sentiment'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive drill-down: Show reviews for selected topic
    st.subheader("Explore Reviews by Topic")
    selected_topic = st.selectbox("Select Topic", range(num_topics))
    
    topic_reviews = df[df['topic'] == selected_topic]
    
    st.write(f"**Top words in Topic {selected_topic}:** {', '.join(topic_words[selected_topic])}")
    st.write(f"**Number of reviews:** {len(topic_reviews)}")
    
    # Show sample reviews for the selected topic
    for i, row in topic_reviews.head(5).iterrows():
        st.write(f"**Rating: {row['rating']} | Sentiment: {row['sentiment']}**")
        st.write(row['review_text'])
        st.write("---")

# Trend Analysis Section
elif options == "Trend Analysis":
    st.header("📈 Trend Analysis")
    
    # Aggregate data by date
    df['date'] = pd.to_datetime(df['date'])
    daily_data = df.groupby(df['date'].dt.date).agg({
        'rating': 'mean',
        'sentiment_score': 'mean',
        'review_text': 'count'
    }).reset_index()
    daily_data.columns = ['date', 'avg_rating', 'avg_sentiment', 'review_count']
    
    # Detect anomalies (spikes in negative sentiment)
    daily_data['neg_sentiment_anomaly'] = False
    sentiment_mean = daily_data['avg_sentiment'].mean()
    sentiment_std = daily_data['avg_sentiment'].std()
    
    # Mark dates with sentiment significantly lower than average
    daily_data.loc[daily_data['avg_sentiment'] < (sentiment_mean - 2 * sentiment_std), 'neg_sentiment_anomaly'] = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Rating Over Time")
        fig = px.line(daily_data, x='date', y='avg_rating', 
                      title="Average Daily Rating",
                      labels={'date': 'Date', 'avg_rating': 'Average Rating'})
        
        # Add markers for anomalies
        anomalies = daily_data[daily_data['neg_sentiment_anomaly']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['date'], 
                y=anomalies['avg_rating'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Anomaly'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Sentiment Over Time")
        fig = px.line(daily_data, x='date', y='avg_sentiment', 
                      title="Average Daily Sentiment Score",
                      labels={'date': 'Date', 'avg_sentiment': 'Average Sentiment'})
        
        # Add markers for anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['date'], 
                y=anomalies['avg_sentiment'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Anomaly'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Review Volume Over Time")
    fig = px.line(daily_data, x='date', y='review_count', 
                  title="Daily Review Volume",
                  labels={'date': 'Date', 'review_count': 'Number of Reviews'})
    
    # Add markers for anomalies
    if not anomalies.empty:
        # Get review count for anomaly dates
        anomaly_counts = daily_data[daily_data['neg_sentiment_anomaly']]['review_count']
        fig.add_trace(go.Scatter(
            x=anomalies['date'], 
            y=anomaly_counts,
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Anomaly'
        ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show anomaly details
    if not anomalies.empty:
        st.subheader("🚨 Sentiment Anomalies Detected")
        
        for _, row in anomalies.iterrows():
            st.write(f"**Date: {row['date']}**")
            st.write(f"Average Sentiment: {row['avg_sentiment']:.3f} (Significantly below average)")
            
            # Show sample reviews from anomaly date
            anomaly_reviews = df[pd.to_datetime(df['date']).dt.date == row['date']]
            st.write(f"Number of reviews: {len(anomaly_reviews)}")
            
            # Show top negative reviews from that date
            negative_reviews = anomaly_reviews[anomaly_reviews['sentiment'] == 'Negative']
            if not negative_reviews.empty:
                st.write("**Sample negative reviews from this date:**")
                for i, review in negative_reviews.head(3).iterrows():
                    st.write(f"Rating: {review['rating']} - {review['review_text']}")
            st.write("---")
    
    # Customer journey analysis (before vs after release)
    st.subheader("Customer Journey Analysis")
    
    # Simulate version release dates
    version_dates = {
        '1.3.0': datetime(2023, 2, 15),
        '1.3.1': datetime(2023, 3, 20),
        '1.4.0': datetime(2023, 5, 10)
    }
    
    selected_version = st.selectbox("Select Version", list(version_dates.keys()))
    release_date = version_dates[selected_version]
    
    # Define time windows around release
    pre_release_start = release_date - timedelta(days=14)
    pre_release_end = release_date - timedelta(days=1)
    post_release_start = release_date + timedelta(days=1)
    post_release_end = release_date + timedelta(days=14)
    
    # Filter data for these periods
    pre_release_data = df[(df['date'] >= pre_release_start) & (df['date'] <= pre_release_end)]
    post_release_data = df[(df['date'] >= post_release_start) & (df['date'] <= post_release_end)]
    
    if not pre_release_data.empty and not post_release_data.empty:
        # Calculate metrics for both periods
        pre_avg_rating = pre_release_data['rating'].mean()
        post_avg_rating = post_release_data['rating'].mean()
        
        pre_avg_sentiment = pre_release_data['sentiment_score'].mean()
        post_avg_sentiment = post_release_data['sentiment_score'].mean()
        
        # Create comparison chart
        metrics = ['Average Rating', 'Average Sentiment']
        pre_values = [pre_avg_rating, pre_avg_sentiment]
        post_values = [post_avg_rating, post_avg_sentiment]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics,
            y=pre_values,
            name='Before Release',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=metrics,
            y=post_values,
            name='After Release',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title=f'Impact of {selected_version} Release on Customer Satisfaction',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate percentage change
        rating_change = ((post_avg_rating - pre_avg_rating) / pre_avg_rating) * 100
        sentiment_change = ((post_avg_sentiment - pre_avg_sentiment) / pre_avg_sentiment) * 100
        
        st.write(f"**Rating change:** {rating_change:+.1f}%")
        st.write(f"**Sentiment change:** {sentiment_change:+.1f}%")
        
        if rating_change > 0 or sentiment_change > 0:
            st.success("The release had a positive impact on customer satisfaction.")
        else:
            st.error("The release had a negative impact on customer satisfaction.")
    else:
        st.warning("Not enough data available for customer journey analysis.")

# Review Summarization Section
elif options == "Review Summarization":
    st.header("📝 Review Summarization")
    
    st.info("""
    AI-generated summaries of customer feedback help quickly understand key themes and sentiments
    without reading through hundreds of individual reviews.
    """)
    
    # Option to select what to summarize
    summarization_option = st.radio(
        "Select what to summarize:",
        ["By Sentiment", "By Topic", "By Product", "Custom Filter"]
    )
    
    if summarization_option == "By Sentiment":
        selected_sentiment = st.selectbox("Select Sentiment", ['Positive', 'Negative', 'Neutral'])
        reviews_to_summarize = df[df['sentiment'] == selected_sentiment]['review_text'].tolist()
        summary_title = f"Summary of {selected_sentiment} Reviews"
        
    elif summarization_option == "By Topic":
        if 'topic' in df.columns:
            selected_topic = st.selectbox("Select Topic", sorted(df['topic'].unique()))
            reviews_to_summarize = df[df['topic'] == selected_topic]['review_text'].tolist()
            
            # Get top words for the topic
            if 'lda_model' in locals():
                top_words = [word for word, _ in lda_model.show_topic(selected_topic, topn=5)]
                summary_title = f"Summary of Topic {selected_topic} ({', '.join(top_words)})"
            else:
                summary_title = f"Summary of Topic {selected_topic}"
        else:
            st.warning("Topic modeling not performed yet. Please run Topic Modeling first.")
            reviews_to_summarize = []
            summary_title = ""
            
    elif summarization_option == "By Product":
        selected_product = st.selectbox("Select Product", df['product'].unique())
        reviews_to_summarize = df[df['product'] == selected_product]['review_text'].tolist()
        summary_title = f"Summary of Reviews for {selected_product}"
        
    else:  # Custom Filter
        col1, col2 = st.columns(2)
        with col1:
            min_rating = st.slider("Minimum Rating", 1, 5, 3)
            selected_sources = st.multiselect("Sources", df['source'].unique(), default=df['source'].unique())
        with col2:
            date_range = st.date_input("Date Range", 
                                     [df['date'].min().date(), df['date'].max().date()])
        
        filtered_df = df[
            (df['rating'] >= min_rating) & 
            (df['source'].isin(selected_sources)) &
            (df['date'] >= pd.to_datetime(date_range[0])) &
            (df['date'] <= pd.to_datetime(date_range[1]))
        ]
        
        reviews_to_summarize = filtered_df['review_text'].tolist()
        summary_title = f"Summary of Filtered Reviews ({len(reviews_to_summarize)} reviews)"
    
    # Generate summary if we have reviews to summarize
    if reviews_to_summarize:
        # Combine reviews into a single text
        combined_text = " ".join(reviews_to_summarize)
        
        # Let user select summary length
        summary_length = st.slider("Summary Length (sentences)", 2, 10, 3)
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(combined_text, sentences_count=summary_length)
                
                st.subheader(summary_title)
                st.info(summary)
                
                # Show some statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Reviews", len(reviews_to_summarize))
                with col2:
                    avg_rating = np.mean([df[df['review_text'] == r]['rating'].values[0] for r in reviews_to_summarize])
                    st.metric("Average Rating", f"{avg_rating:.1f}")
                with col3:
                    avg_sentiment = np.mean([df[df['review_text'] == r]['sentiment_score'].values[0] for r in reviews_to_summarize])
                    st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                
                # Show sample reviews
                st.subheader("Sample Reviews")
                sample_reviews = np.random.choice(reviews_to_summarize, size=min(5, len(reviews_to_summarize)), replace=False)
                for i, review in enumerate(sample_reviews):
                    rating = df[df['review_text'] == review]['rating'].values[0]
                    sentiment = df[df['review_text'] == review]['sentiment'].values[0]
                    st.write(f"**Rating: {rating} | Sentiment: {sentiment}**")
                    st.write(review)
                    st.write("---")
    else:
        st.warning("No reviews match the selected criteria.")

# Actionable Insights Section
elif options == "Actionable Insights":
    st.header("💡 Actionable Insights")
    
    st.success("""
    Based on the analysis of customer feedback, here are the key insights and recommendations:
    """)
    
    # Insight 1: Sentiment by rating
    st.subheader("🔍 Insight 1: Rating-Sentiment Mismatch")
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
    
    # Find ratings with high negative sentiment
    high_neg_ratings = []
    for rating in rating_sentiment.index:
        if rating_sentiment.loc[rating, 'Negative'] > 30:  # More than 30% negative
            high_neg_ratings.append(rating)
    
    if high_neg_ratings:
        st.warning(f"**Finding**: Ratings {', '.join(map(str, high_neg_ratings))} have unexpectedly high negative sentiment")
        st.write("**Implication**: Users giving these ratings are frustrated with specific aspects despite moderately positive overall ratings")
        st.write("**Recommendation**: Analyze text of these reviews to identify specific pain points that need addressing")
    
    # Insight 2: Common topics in negative reviews
    st.subheader("🔍 Insight 2: Common Themes in Negative Feedback")
    
    if 'topic' in df.columns:
        negative_by_topic = df[df['sentiment'] == 'Negative']['topic'].value_counts().nlargest(3)
        
        if len(negative_by_topic) > 0:
            st.warning(f"**Finding**: Most negative feedback is about Topic {negative_by_topic.index[0]}")
            
            # Get top words for this topic
            if 'lda_model' in locals():
                top_words = [word for word, _ in lda_model.show_topic(negative_by_topic.index[0], topn=5)]
                st.write(f"**Related Keywords**: {', '.join(top_words)}")
            
            st.write("**Implication**: This is the area causing most frustration among users")
            st.write("**Recommendation**: Prioritize improvements in this area to reduce negative feedback")
    
    # Insight 3: Sentiment trends
    st.subheader("🔍 Insight 3: Sentiment Trends Over Time")
    
    # Check if sentiment is improving or worsening
    df['month'] = df['date'].dt.month
    monthly_sentiment = df.groupby('month')['sentiment_score'].mean()
    
    if len(monthly_sentiment) > 1:
        trend = "improving" if monthly_sentiment.iloc[-1] > monthly_sentiment.iloc[-2] else "worsening"
        st.info(f"**Finding**: User sentiment is {trend} in recent months")
        
        if trend == "worsening":
            st.write("**Implication**: Recent changes or market conditions may be negatively impacting user experience")
            st.write("**Recommendation**: Investigate recent updates or competitor moves that might be affecting user satisfaction")
        else:
            st.write("**Implication**: Recent improvements are having a positive impact on user experience")
            st.write("**Recommendation**: Continue with the current improvement strategy and double down on what's working")
    
    # Insight 4: Emotion analysis insights
    st.subheader("🔍 Insight 4: Dominant Emotions in Feedback")
    
    # Extract all emotions and their counts
    all_emotions = {}
    for emotions in df['top_emotions']:
        for emotion, count in emotions.items():
            all_emotions[emotion] = all_emotions.get(emotion, 0) + count
    
    if all_emotions:
        top_emotion = max(all_emotions.items(), key=lambda x: x[1])
        st.info(f"**Finding**: The most common emotion in reviews is '{top_emotion[0]}' ({top_emotion[1]} occurrences)")
        
        if top_emotion[0] in ['anger', 'fear', 'sadness']:
            st.write("**Implication**: Users are experiencing strong negative emotions when using the product")
            st.write("**Recommendation**: Address the root causes of these negative emotions urgently")
        elif top_emotion[0] in ['joy', 'surprise']:
            st.write("**Implication**: Users are having positive emotional experiences with the product")
            st.write("**Recommendation**: Identify what's working well and enhance those aspects")
    
    # Insight 5: Aspect-based insights
    st.subheader("🔍 Insight 5: Key Aspects Needing Improvement")
    
    # Extract aspect sentiment data
    aspect_data = []
    for idx, row in df.iterrows():
        for aspect, sentiment_info in row['aspect_sentiment'].items():
            aspect_data.append({
                'aspect': aspect,
                'sentiment': sentiment_info['sentiment'],
                'score': sentiment_info['score']
            })
    
    if aspect_data:
        aspect_df = pd.DataFrame(aspect_data)
        aspect_sentiment_avg = aspect_df.groupby('aspect')['score'].mean().sort_values()
        
        # Find the worst-performing aspect
        worst_aspect = aspect_sentiment_avg.index[0]
        worst_score = aspect_sentiment_avg.iloc[0]
        
        if worst_score < 0:
            st.warning(f"**Finding**: The worst-performing aspect is '{worst_aspect}' with sentiment score {worst_score:.2f}")
            st.write("**Implication**: This aspect is causing significant user dissatisfaction")
            st.write("**Recommendation**: Prioritize improvements to this aspect in the next product update")
        
        # Find the best-performing aspect
        best_aspect = aspect_sentiment_avg.index[-1]
        best_score = aspect_sentiment_avg.iloc[-1]
        
        if best_score > 0:
            st.success(f"**Finding**: The best-performing aspect is '{best_aspect}' with sentiment score {best_score:.2f}")
            st.write("**Implication**: This aspect is contributing significantly to user satisfaction")
            st.write("**Recommendation**: Highlight this strength in marketing materials and continue to invest in it")
    
    # Summary of recommendations
    st.subheader("🎯 Summary of Recommendations")
    
    recommendations = [
        "Prioritize fixes for the most common topics in negative feedback",
        "Address the aspects with the lowest sentiment scores",
        "Conduct deeper analysis on ratings with sentiment mismatch",
        "Monitor sentiment trends monthly to measure impact of changes",
        "Create a feedback loop to inform users about implemented improvements",
        "Consider A/B testing for features receiving mixed feedback",
        "Leverage positive emotions and aspects in marketing campaigns",
        "Develop a plan to address the root causes of negative emotions"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox label {
        font-weight: bold;
    }
    .insight-positive {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin-bottom: 15px;
    }
    .insight-negative {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin-bottom: 15px;
    }
    .insight-neutral {
        background-color: #e2e3e5;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #6c757d;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 📊 Advanced Customer Feedback NLP Analysis Dashboard")
st.markdown("Built with Streamlit | Using NLTK, VADER, NRCLex, and LDA for text analysis")