import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import joblib
import logomaker
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer
model = joblib.load("Model/naive_bayes_model.pkl")
vectorizer = joblib.load("Model/count_vectorizer.pkl")

# Class mapping
class_mappings = {
    0: "G Protein Coupled Receptors",
    1: "Tyrosine Kinase",
    2: "Tyrosine Phosphatase",
    3: "Synthetase",
    4: "Synthase",
    5: "Ion Channel",
    6: "Transcription Factor"
}

# Function to extract k-mers
def get_kmers(sequence, size=6):
    return [sequence[i:i+size] for i in range(len(sequence)-size+1)]

df = pd.read_table("examples.txt")

examples = df["examples"].tolist()

# Example sequences
example_sequences = {
    "Example 1": examples[0],
    "Example 2": examples[1],
    "Example 3": examples[2]
}

# Page title
st.title("🧬 DNA Sequence Classifier")

# Sidebar - Upload or Select Example
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("Upload DNA Sequence File (.txt)", type=["txt"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Or choose an example sequence:")
selected_example = st.sidebar.radio("Example Sequences", list(example_sequences.keys()))

# Read uploaded or example file
sequence = ""
source = ""

if uploaded_file:
    raw = uploaded_file.read().decode("utf-8")
    sequence = ''.join([line.strip() for line in raw.splitlines() if not line.startswith(">")]).upper()
    source = "uploaded"
elif selected_example:
    sequence = example_sequences[selected_example]
    source = "example"

if sequence:
    st.subheader("📥 Input DNA Sequence")
    st.text_area("Sequence (first 1000 characters shown)", sequence[:1000], height=150)

    # Base Distribution
    st.subheader("🔬 Nucleotide Distribution")
    base_counts = Counter(sequence)
    bases = ['A', 'T', 'G', 'C']
    counts = [base_counts.get(base, 0) for base in bases]
    fig1, ax1 = plt.subplots()
    ax1.bar(bases, counts, color=['green', 'red', 'blue', 'orange'])
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Top k-mers
    st.subheader("🔠 Top 10 6-mers")
    kmers = get_kmers(sequence, size=6)
    top_kmers = Counter(kmers).most_common(10)
    df_top = pd.DataFrame(top_kmers, columns=["6-mer", "Count"])
    st.dataframe(df_top)

    # Prediction
    st.subheader("🤖 Predicted Class")
    kmers_text = ' '.join(kmers)
    vectorized = vectorizer.transform([kmers_text])
    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    st.markdown(f"### 🧬 Class: `{class_mappings[pred]}`")
    st.markdown(f"Confidence: `{proba[pred]*100:.2f}%`")
else:
    st.info("Please upload a file or select an example to begin.")
