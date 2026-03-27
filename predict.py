import fitz
import joblib

from resume_preprocessing import preprocess_text
from skill_loader import load_job_skills

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 🔹 Load trained model (pipeline)
model = joblib.load("model.pkl")

# 🔹 Load SBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Load job dataset
job_skills = load_job_skills(
    r"C:\Users\Gali Supriya\Desktop\IT Job Roles & Skills Dataset\IT_Job_Roles_Skills.csv"
)


# 🔹 Get all skills (NO HARDCODING)
def get_all_skills(skill_dict):
    skills = set()
    for s_list in skill_dict.values():
        for s in s_list:
            skills.add(s.lower())
    return list(skills)


all_skills = get_all_skills(job_skills)

# 🔥 Precompute skill embeddings (IMPORTANT)
skill_embeddings = sbert.encode(all_skills)


# 🔹 Extract text from PDF
def extract_text(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


# 🔥 SEMANTIC SKILL EXTRACTION
def extract_skills(text):
    sentences = text.split(".")
    extracted_skills = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Encode sentence
        sent_embedding = sbert.encode([sentence])

        # Compare with all skills
        similarities = cosine_similarity(sent_embedding, skill_embeddings)[0]

        # Get top matches
        top_indices = similarities.argsort()[-5:]

        for idx in top_indices:
            if similarities[idx] > 0.5:  # threshold
                extracted_skills.add(all_skills[idx].capitalize())

    # Limit output (avoid noise)
    return list(extracted_skills)[:20]


# 🔹 Filter important skills (NO HARDCODING using frequency)
from collections import Counter

def get_top_skills(skill_dict, top_n=100):
    counter = Counter()
    for skills in skill_dict.values():
        for skill in skills:
            counter[skill.lower()] += 1
    return set([skill for skill, _ in counter.most_common(top_n)])


top_skills = get_top_skills(job_skills)


def filter_technical(skills):
    return [s for s in skills if s.lower() in top_skills]


# 🔹 Role matching using SBERT
def get_required_skills(predicted_role, job_skills):
    roles = list(job_skills.keys())

    role_embeddings = sbert.encode(roles)
    pred_embedding = sbert.encode([predicted_role])

    similarities = cosine_similarity(pred_embedding, role_embeddings)[0]

    # Top 3 matches (less noise)
    top_indices = similarities.argsort()[-3:][::-1]

    required = []

    for idx in top_indices:
        required.extend(job_skills[roles[idx]])

    return list(set(required))


# 🔹 Skill recommendation
def recommend_learning(skills):
    resources = {}
    for skill in skills:
        resources[skill] = f"Learn {skill} via Coursera / Udemy"
    return resources


# 🔹 Main function
def analyze_resume(file_path):
    text = extract_text(file_path)

    cleaned = preprocess_text(text)

    # Predict role
    role = model.predict([cleaned])[0]

    # Extract skills (semantic)
    user_skills = extract_skills(text)

    # Get required skills
    required_skills = get_required_skills(role, job_skills)

    # Filter relevant skills
    required_skills = filter_technical(required_skills)

    # Skill gap
    missing_skills = list(set(required_skills) - set(user_skills))

    # Recommendations
    recommendations = recommend_learning(missing_skills)

    return role, user_skills, missing_skills, recommendations


# 🔹 Run
if __name__ == "__main__":
    file_path = r"C:\Users\Gali Supriya\Downloads\Supriya_Resume.pdf"

    role, skills, missing, recs = analyze_resume(file_path)

    print("\nPredicted Role:", role)
    print("Your Skills:", skills)
    print("Missing Skills:", missing)

    print("\nRecommended Learning:")
    for skill, res in recs.items():
        print(f"{skill} → {res}")