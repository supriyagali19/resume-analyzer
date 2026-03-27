import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])


def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


def preprocess_dataset(df):
    df["cleaned_resume"] = df["Resume"].apply(preprocess_text)
    return df