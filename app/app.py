"""
Gradio web app for Kirundi news topic classification.

The fine-tuned AfroXLMR model is loaded from the Hugging Face Hub (MODEL_ID) and
the app returns the predicted topic with class probabilities.

Run locally:   python app/app.py
On a Space:    set MODEL_ID in the Space settings (Variables), or edit the default below.
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "YOUR_HF_USERNAME/kirundi-news-afroxlmr")
MAX_LEN = 256            # same truncation length used in training (first 256 tokens)
MIN_WORDS = 5            # refuse near-empty input
LOW_CONFIDENCE = 0.50    # below this we warn the user

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]


def predict(text: str):
    text = (text or "").strip()
    if len(text.split()) < MIN_WORDS:
        return {}, "Please paste a longer Kirundi text (at least a full sentence)."
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top = max(scores, key=scores.get)
    note = f"Predicted topic: **{top}** ({scores[top]:.0%} confidence)."
    if scores[top] < LOW_CONFIDENCE:
        note += " The model is not confident: the text may be off-topic, very short, or in another language."
    if len(tokenizer(text)["input_ids"]) > MAX_LEN:
        note += f" (Only the first {MAX_LEN} tokens were used.)"
    return scores, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, label="Kirundi news text",
                      placeholder="Paste a Kirundi news article or paragraph here..."),
    outputs=[gr.Label(num_top_classes=5, label="Topic probabilities"),
             gr.Markdown(label="Result")],
    title="Kirundi News Topic Classifier",
    description=("Fine-tuned AfroXLMR on the KIRNEWS corpus (12 topics: politics, sport, economy, health, "
                 "entertainment, history, technology, culture, religion, environment, education, "
                 "relationship). Rare topics (history, religion, culture) have very few training "
                 "articles, so predictions for them are less reliable."),
    examples=[],   # TODO: add 3-4 short Kirundi texts of your own (e.g. from the test split)
)

if __name__ == "__main__":
    demo.launch()
