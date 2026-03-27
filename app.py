import streamlit as st
import tempfile

from predict import analyze_resume


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get role prediction + skill gap analysis 🚀")

# Upload file
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("Analyze Resume"):

        with st.spinner("Analyzing..."):

            role, skills, missing, recs = analyze_resume(temp_path)

        st.success("Analysis Complete!")

        # 🎯 Role
        st.subheader("🎯 Predicted Role")
        st.write(f"👉 **{role}**")

        # 🧠 Skills
        st.subheader("🧠 Your Skills")
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No skills detected")

        # ⚠️ Missing Skills
        st.subheader("⚠️ Missing Skills")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("No major skill gaps 🎉")

        # 📚 Recommendations
        st.subheader("📚 Learning Recommendations")

        if recs:
            for skill, res in recs.items():
                st.write(f"🔹 {skill}: {res}")
        else:
            st.write("You're all set! 🚀")