import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np

# Emotion labels
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Cache the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "roberta-goemotions-model"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("GoEmotions - Emotion Detection")

# Input
text = st.text_area("Enter a sentence:")
top_k = st.slider("Number of top emotions to display", min_value=1, max_value=10, value=5)

if st.button("Analyze Emotions"):
    text = text.strip().replace('\n', ' ').replace('\r', ' ')

    if text == "":
        st.warning("Please enter some text.")
    else:
        try:
            # Tokenize and keep on CPU
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

            with torch.no_grad():
                outputs = model(**inputs)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            sorted_indices = np.argsort(-probs)

            st.write(f"### Top {top_k} Emotions:")
            for idx in sorted_indices[:top_k]:
                prob_percent = probs[idx] * 100
                st.write(f"**{label_names[idx].capitalize()}** ({prob_percent:.2f}%)")
                st.progress(float(probs[idx]))

        except Exception as e:
            st.error(f"Error during analysis: {e}")
