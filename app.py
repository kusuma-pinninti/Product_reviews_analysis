import streamlit as st
import joblib
import gdown
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
from collections import Counter
import spacy
import google.generativeai as genai  # For Gemini API

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load spaCy model (small English model)
nlp = spacy.load("en_core_web_sm")

# Google Drive file IDs (replace with your actual file IDs)
MODEL_FILE_ID = "1XEaBunWsBdU9-Vjp9pE8nffR4HxDrpY3"
VECTORIZER_FILE_ID = "1eqhrs7kp4X0DXskiI4rBMS5V3m0eZN8j"

# File paths
model_path = "random_forest_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# Function to download files from Google Drive
@st.cache_data
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Download files
download_file(MODEL_FILE_ID, model_path)
download_file(VECTORIZER_FILE_ID, vectorizer_path)

# Load Model and Vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Set up Gemini API
GEMINI_API_KEY = "AIzaSyDKl3pN0X1sIA6RCAu1kjb1c8xuKt9Hylc" 
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel("gemini-pro")

# Proxy settings (modify with your proxy server if needed)
PROXY = {
    "http": "189.240.60.169:9090",
    "https": "13.36.113.81:3128"
}

# Function to preprocess text
def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Function to scrape product info from Amazon
def scrape_amazon_product_info(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        response = requests.get(url, headers=headers, proxies=PROXY)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract product name
        product_name = soup.find('span', {'id': 'productTitle'})
        product_name = product_name.text.strip() if product_name else "Product Name Not Found"

        # Extract product image URL
        image_tag = soup.find('img', {'id': 'landingImage'}) or soup.find('img', {'class': 'a-dynamic-image'})
        image_url = image_tag['src'] if image_tag else None

        # Extract reviews
        reviews = [review.text for review in soup.find_all('span', {'data-hook': 'review-body'})]
        return product_name, image_url, reviews[:10]  # Limit to first 10 reviews
    except Exception as e:
        st.error(f"Error scraping Amazon product info: {e}")
        return None, None, None

def analyze_reviews(reviews):
    fake_count, genuine_count = 0, 0
    feature_sentiments = {}

    for review in reviews:
        processed = text_preprocess(review)
        if processed:
            tfidf_review = vectorizer.transform([processed])
            prediction = model.predict(tfidf_review)[0]
           # st.write(f"Review: {review}")  # Debugging: Print the review
            #st.write(f"Prediction: {prediction}")  # Debugging: Print the prediction

            if prediction == "OR":
                genuine_count += 1
            else:
                fake_count += 1

            # Identify adjective-noun pairs for features
            doc = nlp(review)
            for token in doc:
                if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                    feature = token.head.text
                    sentiment = 1 if token.text in ["good", "great", "excellent", "love", "amazing"] else -1 if token.text in ["bad", "poor", "worst", "disappointed", "awful"] else 0
                    if sentiment != 0:
                        feature_sentiments[feature] = feature_sentiments.get(feature, 0) + sentiment

    total = len(reviews)
    fake_percentage = (fake_count / total) * 100 if total > 0 else 0
    genuine_percentage = 100 - fake_percentage

    # Separate pros and cons based on sentiment scores
    pros = [f for f, score in sorted(feature_sentiments.items(), key=lambda x: x[1], reverse=True) if score > 0][:5]
    cons = [f for f, score in sorted(feature_sentiments.items(), key=lambda x: x[1]) if score < 0][:5]

    avg_rating = round((genuine_percentage / 100) * 5, 2)

    return {
        'fake_percentage': fake_percentage,
        'genuine_percentage': genuine_percentage,
        'average_rating': avg_rating,
        'pros': pros,
        'cons': cons
    }

# Function to analyze reviews using Gemini API
def analyze_reviews_with_gemini(reviews):
    try:
        # Combine all reviews into a single text
        combined_reviews = " ".join(reviews)

        # Generate a prompt for Gemini
        prompt = f"Analyze the following product reviews and provide a summary of pros and cons:\n\n{combined_reviews}\n\nPros:\n1. \n\nCons:\n1. "

        # Call Gemini API
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing reviews with Gemini API: {e}")
        return None

# Custom CSS to make the app wider, add margins, and style the footer
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding: 2rem;
        margin: auto;
    }
    .footer {
        padding: 20px;
        background-color: #f0f2f6;
        border-top: 2px solid #ddd;
        text-align: center;
        font-family: Arial, sans-serif;
        color: #333;
        margin-top: 30px;
    }
    .footer h3 {
        margin-bottom: 10px;
        color: #555;
    }
    .footer p {
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🛒 Fake Review Detection System")
st.write("Enter a review or a product URL to check for fake reviews.")

# Create two columns for side-by-side layout
col1, col2 = st.columns(2)

# Single Review Analysis (Left Side)
with col1:
    st.subheader("Single Review Analysis")
    single_review = st.text_area("Enter a single review:", height=100)  # Adjusted height

# Product Review Analysis (Right Side)
with col2:
    st.subheader("Product Review Analysis")
    product_url = st.text_input("Enter product URL (Amazon):")

# Buttons for analysis (placed side by side in their own sections)
button_col1, button_col2 = st.columns(2)

with button_col1:
    if st.button("Analyze Single Review"):
        if single_review.strip():
            processed_review = text_preprocess(single_review)
            tfidf_review = vectorizer.transform([processed_review])
            prediction = model.predict(tfidf_review)[0]

            if prediction == "OR":
                st.success("✅ Genuine Review")
            else:
                st.error("⚠ Fake Review")
        else:
            st.warning("Please enter a review to analyze.")

with button_col2:
    if st.button("Analyze Product Reviews"):
        if product_url.strip():
            if "amazon" in product_url.lower():
                product_name, product_image, reviews = scrape_amazon_product_info(product_url)
                if product_name:
                    if product_image:
                        st.image(product_image, caption=product_name, use_column_width=False, width=300)  # Decrease image size
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{product_name}</h2>", unsafe_allow_html=True)

                    if reviews:
                        # Analyze reviews using the model
                        results = analyze_reviews(reviews)
                        st.subheader("🔍 Analysis Results")
                        st.write(f"*Fake Reviews*: {results['fake_percentage']:.2f}%")
                        st.write(f"*Genuine Reviews*: {results['genuine_percentage']:.2f}%")
                        st.write(f"*Estimated Rating*: {results['average_rating']}/5")

                        # Analyze reviews using Gemini API for better pros and cons
                        st.subheader("🤖 Gemini Analysis")
                        gemini_result = analyze_reviews_with_gemini(reviews)
                        if gemini_result:
                            st.write(gemini_result)
                        else:
                            st.warning("Unable to analyze reviews with Gemini API.")
                    else:
                        st.warning("Unable to retrieve reviews. Check the URL or try another product.")
                else:
                    st.error("Failed to scrape product information. Please check the URL.")
            else:
                st.error("Unsupported website. Please provide a valid Amazon URL.")
        else:
            st.warning("Please enter a valid product URL.")

# Footer with Project Contributors and Guide in a row
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <div style="display: flex; justify-content: space-around;">
            <div>
                <h3>Project Contributors</h3>
                <p><strong>Pinninti Kusuma</strong></p>
                <p><strong>Polamarasetti Vivek Vardhan</strong></p>
                <p><strong>Pinninti Sai Manikanta</strong></p>
                <p><strong>Potabatuula Arya</strong></p>
            </div>
            <div>
                <h3>Guided by</h3>
                <p><strong>Dr. N.V.S Lakshmipathi Raju</strong></p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
