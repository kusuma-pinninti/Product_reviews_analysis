import streamlit as st
import joblib
import gdown
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import spacy
import google.generativeai as genai  # For Gemini API

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Google Drive file IDs
MODEL_FILE_ID = "1XEaBunWsBdU9-Vjp9pE8nffR4HxDrpY3"
VECTORIZER_FILE_ID = "1eqhrs7kp4X0DXskiI4rBMS5V3m0eZN8j"

# File paths
model_path = "random_forest_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

@st.cache_data
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

download_file(MODEL_FILE_ID, model_path)
download_file(VECTORIZER_FILE_ID, vectorizer_path)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Gemini API setup
GEMINI_API_KEY = "AIzaSyDKl3pN0X1sIA6RCAu1kjb1c8xuKt9Hylc"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

def scrape_flipkart_product_info(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    try:
        title = driver.find_element(By.CLASS_NAME, "B_NuCI").text
        image = driver.find_element(By.CLASS_NAME, "_396cs4").get_attribute("src")
        reviews = [elem.text for elem in driver.find_elements(By.CLASS_NAME, "t-ZTKy")][:10]
        driver.quit()
        return title, image, reviews
    except:
        driver.quit()
        return None, None, None

def scrape_amazon_product_info(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        product_name = soup.find('span', {'id': 'productTitle'}).text.strip()
        image = soup.find('img', {'id': 'landingImage'})['src']
        reviews = [r.text for r in soup.find_all('span', {'data-hook': 'review-body'})][:10]
        return product_name, image, reviews
    except:
        return None, None, None

def analyze_reviews(reviews):
    fake_count, genuine_count = 0, 0
    for review in reviews:
        processed = text_preprocess(review)
        tfidf_review = vectorizer.transform([processed])
        prediction = model.predict(tfidf_review)[0]
        if prediction == "OR":
            genuine_count += 1
        else:
            fake_count += 1
    fake_percentage = (fake_count / len(reviews)) * 100 if reviews else 0
    genuine_percentage = 100 - fake_percentage
    avg_rating = round((genuine_percentage / 100) * 5, 2)
    return {'fake_percentage': fake_percentage, 'genuine_percentage': genuine_percentage, 'average_rating': avg_rating}

st.title("🛒 Fake Review Detection System")
product_url = st.text_input("Enter product URL (Amazon or Flipkart):")

if st.button("Analyze Product Reviews"):
    if "amazon" in product_url.lower():
        name, image, reviews = scrape_amazon_product_info(product_url)
    elif "flipkart" in product_url.lower():
        name, image, reviews = scrape_flipkart_product_info(product_url)
    else:
        st.error("Unsupported website. Please use Amazon or Flipkart.")
        name, image, reviews = None, None, None
    
    if name:
        st.image(image, caption=name, use_column_width=True)
        results = analyze_reviews(reviews)
        st.write(f"Fake Reviews: {results['fake_percentage']:.2f}%")
        st.write(f"Genuine Reviews: {results['genuine_percentage']:.2f}%")
        st.write(f"Estimated Rating: {results['average_rating']}/5")
    else:
        st.error("Failed to retrieve product details. Try another URL.")
