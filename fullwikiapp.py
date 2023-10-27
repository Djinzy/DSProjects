#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
try:
    from bs4 import BeautifulSoup
except :
    from BeautifulSoup import BeautifulSoup 
import requests
import nltk
from string import punctuation

st.title("Wikipedia ChatBot")

class ChatBot:
    def __init__(self):
        self.end_chat = False
        self.got_topic = False
        self.title = None
        self.text_data = []
        self.display_more = False  # Flag to control the "more" button display
        self.last_displayed_index = 0  # Track the last displayed paragraph index

    def preprocess_input(self, text):
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', punctuation))
        words = nltk.word_tokenize(text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def chat(self):
        user_input_topic = st.text_input("Enter a topic of interest:")
        user_input_query = st.text_input("Ask your question:")

        if user_input_topic and not self.got_topic:
            self.scrape_wiki(user_input_topic)

        if user_input_query:
            user_input_query = self.preprocess_input(user_input_query)
            if self.got_topic:
                response = self.search_wiki(user_input_query)
                st.write("ChatBot >>  " + response)
                self.display_more = True  # Enable the "more" button

        # Display the "more" button if the flag is set to True
        if self.display_more:
            if st.button("More"):
                self.display_more_info()

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
            st.write('ChatBot >>  Topic is "Wikipedia: {}". Let\'s chat!'.format(self.title))
        except Exception as e:
            st.write('ChatBot >>  Error: {}. Please choose another topic!'.format(e))

    def search_wiki(self, query):
        query = query.lower()
        for text in self.text_data:
            if query in text.lower():
                return text
        return "I couldn't find information related to your query."

    def display_more_info(self):
        if self.got_topic and self.last_displayed_index < len(self.text_data) - 1:
            # Display one additional paragraph from Wikipedia about the same topic.
            self.last_displayed_index += 1
            additional_info = self.text_data[self.last_displayed_index]
            st.write("ChatBot >> Here is more information about the topic:")
            st.write(additional_info)

def main():
    chatbot = ChatBot()
    chatbot.chat()

if __name__ == "__main__":
    main()


# In[ ]:




