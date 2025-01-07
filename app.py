import streamlit as st
import json
import random
import os

# First set up the page config
st.set_page_config(page_title="Sustainability Assistant", page_icon="🌱")

# Handle NLTK installation first, before other imports
try:
    import nltk
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    
    # Download required NLTK data
    for package in ['stopwords', 'wordnet']:
        try:
            nltk.download(package, quiet=True, download_dir=nltk_data_dir)
        except Exception as e:
            st.error(f"Failed to download NLTK package {package}: {str(e)}")
            st.stop()
            
    # Only import NLTK components after successful download
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception as e:
    st.error(f"Failed to initialize NLTK: {str(e)}")
    st.stop()

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    try:
        return (
            WordNetLemmatizer(),
            set(stopwords.words('english')),
            RegexpTokenizer(r'\w+')
        )
    except Exception as e:
        st.error(f"Error initializing NLTK components: {str(e)}")
        st.stop()

lemmatizer, stop_words, tokenizer = initialize_nltk()

# Load knowledge base
@st.cache_data
def load_knowledge_base():
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return {}

knowledge_base = load_knowledge_base()

def preprocess_text(text):
    try:
        tokens = tokenizer.tokenize(text.lower())
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    except Exception as e:
        st.error(f"Error preprocessing text: {str(e)}")
        return []

def find_best_match(user_input):
    try:
        user_tokens = set(preprocess_text(user_input))
        best_match = None
        highest_similarity = 0
        
        for intent, data in knowledge_base.items():
            for pattern in data['patterns']:
                pattern_tokens = set(preprocess_text(pattern))
                if not pattern_tokens:
                    continue
                
                intersection = len(user_tokens.intersection(pattern_tokens))
                union = len(user_tokens.union(pattern_tokens))
                
                if union == 0:
                    continue
                    
                similarity = intersection / union
                if similarity > highest_similarity and similarity > 0.2:
                    highest_similarity = similarity
                    best_match = intent
        
        return best_match
    except Exception as e:
        st.error(f"Error finding best match: {str(e)}")
        return None

# Display title and description
st.title("🌱 Sustainability Assistant")
st.markdown("Ask me anything about sustainability, environmental practices, and eco-friendly living!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about sustainability?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    intent = find_best_match(prompt)
    if intent and intent in knowledge_base:
        response = random.choice(knowledge_base[intent]['responses'])
    else:
        response = "I'm not sure how to respond to that. Could you rephrase your question about sustainability?"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})