import pandas as pd


def load_job_skills(file_path):
    df = pd.read_csv(file_path, encoding="latin1")

    skill_dict = {}

    for _, row in df.iterrows():
        role = str(row["Job Title"]).upper()
        skills = str(row["Skills"])

        skill_list = [s.strip() for s in skills.split(",") if s.strip()]

        skill_dict[role] = skill_list

    return skill_dict