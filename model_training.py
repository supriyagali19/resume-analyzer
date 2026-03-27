import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_loader import load_resume_data
from resume_preprocessing import preprocess_dataset


def train_and_save_model():

    path = "C:/Users/Gali Supriya/Desktop/resume_classification/data/data"

    df = load_resume_data(path)
    df = preprocess_dataset(df)

    X = df["cleaned_resume"]
    y = df["Category"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
        ("clf", LinearSVC())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, "model.pkl")

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    train_and_save_model()