#!/usr/bin/env python
# coding: utf-8

# In[9]:


##Importing relevant Libraries

import streamlit as st
from bs4 import BeautifulSoup
import requests
import nltk
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Define custom CSS styles
custom_css = """
<style>
/* Style the title */
h1 {
    font-size: 2.5em;
    color: #333; /* Dark gray color */
    text-shadow: 2px 2px 3px #888; /* Text shadow for a modern look */
}

/* Style the button with a metallic theme */
button {
    background-color: #888; /* Metallic color */
    color: #fff; /* White text color */
    border: none;
    padding: 10px 20px;
    margin: 10px 0;
    border-radius: 5px;
    cursor: pointer;
    box-shadow: 2px 2px 5px #444; /* Button shadow for a metallic effect */
}

/* Style the input fields */
input {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    margin: 10px 0;
}

/* Style the chatbot responses */
p {
    color: #333; /* Dark gray text color */
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

/* Style the topic and question buttons */
.stTextInput, .stButton {
    background-color: #f0f0f0; /* Subtle gray background color */
    color: #333; /* Dark gray text color */
    border: none;
    padding: 10px 20px;
    margin: 10px 0;
    border-radius: 5px;
    cursor: pointer;
}
</style>
"""

# Apply custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Wikipedia ChatBot")

class ChatBot:
    
# Initializing the chatbot's attributes.
    def __init__(self):
        self.end_chat = False
        self.got_topic = False
        self.title = None
        self.text_data = []
        self.display_more = False
        self.last_displayed_index = 0
        self.sentences = []
        self.displayed_paragraphs = set()
        
# Preprocessing user input to standardize and clean it.
    def preprocess_input(self, text):
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', punctuation))
        words = nltk.word_tokenize(text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
# Scraping and storing Wikipedia content for a given topic.
    def scrape_wiki(self, topic):
        topic = topic.lower().strip().capitalize().split(' ')
        topic = '_'.join(topic)
        try:
            link = 'https://en.wikipedia.org/wiki/' + topic
            data = requests.get(link).content
            soup = BeautifulSoup(data, 'html.parser')
            p_data = soup.findAll('p')
            self.text_data = [p.get_text() for p in p_data]
            self.title = soup.find('h1').string
            self.got_topic = True
            st.write('ChatBot  ►►   Topic is "Wikipedia: {}". Let\'s chat!'.format(self.title))
            self.sentences = self.text_data  # Store Wikipedia paragraphs for later use
        except Exception as e:
            st.write('ChatBot  ►►   Error: {}. Please choose another topic!'.format(e))
            
# Searching Wikipedia content for relevant information based on user queries.
    def search_wiki(self, query):
        query = self.preprocess_input(query)
        matching_paragraphs = []
        
        for i, text in enumerate(self.text_data):
            if any(keyword in text for keyword in query.split()):
                matching_paragraphs.append((i, text))

        if matching_paragraphs:
            # Ranking the matching paragraphs using TF-IDF
            ranked_paragraphs = self.rank_paragraphs(query, matching_paragraphs)
            # Displaying the top-ranked paragraph
            top_paragraph = ranked_paragraphs[0][1]
            return top_paragraph

        return "I couldn't find information related to your query."
    
# Using TF-IDF to rank matching paragraphs for relevance to user queries.
    def rank_paragraphs(self, query, matching_paragraphs):
        # Use TF-IDF to rank matching paragraphs
        corpus = [text for _, text in matching_paragraphs]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_tfidf = vectorizer.transform([query])

        # Calculating cosine similarity between the query and paragraphs
        similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)       
        # Pairing each paragraph with its similarity score
        ranked_paragraphs = list(zip(range(len(matching_paragraphs)), similarity_scores))
        # Sorting the paragraphs by descending similarity score
        ranked_paragraphs = sorted(ranked_paragraphs, key=lambda x: x[1], reverse=True)
        
        return [(matching_paragraphs[i][0], matching_paragraphs[i][1]) for i, _ in ranked_paragraphs]
    
# Generate a response to the user's query.
    def respond(self, user_query):
        # Use the TF-IDF model to respond to the user query
        response = self.search_wiki(user_query)
        return response
    
# Handling the chat interaction with the user, including topic selection and query handling.
    def chat(self):
        user_input_topic = st.text_input("Enter a topic of interest:")
        user_input_query = st.text_input("Ask your question:")

        if user_input_topic and not self.got_topic:
            self.scrape_wiki(user_input_topic)

        if user_input_query:
            user_input_query = self.preprocess_input(user_input_query)
            if self.got_topic:
                response = self.search_wiki(user_input_query)
                if response:
                    st.write("ChatBot  ►►   " + response)
                else:
                    # If not found in Wikipedia, use the TF-IDF model to respond
                    response = self.respond(user_input_query)
                    st.write("ChatBot  ►►   " + response)
                self.display_more = True

                
        if self.display_more:
            if st.button("More"):
                self.display_more_info()
                
# Displaying additional information from Wikipedia while avoiding repetitions.
    def display_more_info(self):
        if self.got_topic and self.last_displayed_index < len(self.text_data) - 1:
            self.last_displayed_index += 1
            while self.last_displayed_index in self.displayed_paragraphs:
                self.last_displayed_index += 1

            if self.last_displayed_index < len(self.text_data):
                additional_info = self.text_data[self.last_displayed_index]
                additional_info = re.sub(r'\[.*?\]', '', additional_info)
                st.write("ChatBot  ►►  Here is more information about the topic:")
                st.write(additional_info)
            
                # Mark the paragraph as displayed
                self.displayed_paragraphs.add(self.last_displayed_index)
                
# Deployment Script

def main():
    chatbot = ChatBot()
    chatbot.chat()

if __name__ == "__main__":
    main()


# In[ ]:




