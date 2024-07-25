import openai
import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import crud, models
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Define a semaphore to limit the number of concurrent threads
semaphore = threading.Semaphore(5)      # Adjust the number as needed

# Load the text generation pipeline from Hugging Face
generator = pipeline('text-generation', model='gpt2', truncation=True, pad_token_id=50256)
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def generate_content(db: Session, topic: str) -> str:
    with semaphore:
        search_term = crud.get_search_term(db, topic)
        if not search_term:
            search_term = crud.create_search_term(db, topic)
        # response = openai.ChatCompletion.create(
        #     model = "gpt-3.5-turbo",
        #     messages = [
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": f"Write a detailed article about {topic}"},
        #     ]
        # )
        # generated_text = response.choices[0].message['content'].strip()
        
        prompt = f"Write for me a detailed article about {topic}"
        response = generator(prompt, max_length=500, num_return_sequences=1, truncation=True)
        generated_text = response[0]['generated_text'].strip()
        
        crud.create_generated_content(db, generated_text, search_term.id)
        return generated_text
        
def analyze_content(db: Session, content: str):
    with semaphore:
        search_term = crud.get_search_term(db, content)
    if not search_term:
        search_term = crud.create_search_term(db, content)
        
    readability = get_readability_score(content)
    sentiment = get_sentiment_analysis(content)
    crud.create_sentiment_analysis(db, readability, sentiment, search_term.id)
    return readability, sentiment
    
def get_readability_score(content: str) -> str:
    return "Readability Score: Good"         # ---- > IF available -> replace with actual readability api call


def get_sentiment_analysis(content: str) -> str:
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": f"Analyze the sentiment of the following text: \n\n{content}\n\nIs the sentiment positive,  negative, or neutral?"},
    #     ],
    #     max_tokens=10
    # )
    # return response.choices[0].message['content'].strip()
    # Use Hugging Face's sentiment-analysis pipeline
    result = sentiment_analyzer(content)
    sentiment = result[0]['label']
    return sentiment
