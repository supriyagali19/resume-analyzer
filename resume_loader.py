import os
import pandas as pd
import fitz


def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except:
        pass
    return text


def load_resume_data(data_path):
    texts = []
    labels = []

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)

        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)

                if file.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)

                    if text.strip():
                        texts.append(text)
                        labels.append(category)

    return pd.DataFrame({
        "Resume": texts,
        "Category": labels
    })