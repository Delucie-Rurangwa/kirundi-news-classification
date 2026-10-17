"""
Kirundi News Topic Classifier - Gradio web app.

The fine-tuned AfroXLMR model is loaded from the Hugging Face Hub (MODEL_ID).
Works on a Hugging Face ZeroGPU Space (the prediction function is decorated with
@spaces.GPU), on CPU hardware, and on a laptop (`python app.py`).
"""
import html
import os

try:                                   # must be imported before torch on ZeroGPU
    import spaces
    gpu = spaces.GPU
except Exception:                      # local run or CPU hardware: no-op decorator
    def gpu(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "delucie/kirundi-news-afroxlmr")
GITHUB_URL = "https://github.com/Delucie-Rurangwa/kirundi-news-classification"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MAX_LEN = 256            # same truncation length as in training (first 256 tokens)
MIN_WORDS = 5            # refuse near-empty input
EXAMPLES = []            # optional: add short Kirundi texts of your own, e.g. ["Umutwe w'inkuru ..."]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
if torch.cuda.is_available():          # on ZeroGPU the GPU is attached only while predict() runs
    model.to("cuda")
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]

ICONS = {"politics": "🏛️", "sport": "⚽", "economy": "💰", "health": "🏥",
         "entertainment": "🎭", "history": "📜", "technology": "💻", "culture": "🎨",
         "religion": "⛪", "environment": "🌍", "education": "🎓", "relationship": "❤️"}


# ----------------------------------------------------------------- HTML pieces
def card(inner: str, border: str = "#e5e7eb") -> str:
    return ('<div style="background:#ffffff;border:1px solid %s;border-radius:16px;padding:22px;'
            'color:#1f2937;box-shadow:0 2px 8px rgba(15,23,42,.08)">%s</div>' % (border, inner))


HEADER = """
<div style="background:linear-gradient(135deg,#4f46e5 0%,#0891b2 100%);border-radius:18px;
            padding:26px 30px;color:#ffffff;margin-bottom:6px">
  <div style="font-size:30px;font-weight:800;letter-spacing:-0.5px">📰 Kirundi News Topic Classifier</div>
  <div style="font-size:16px;margin-top:8px;opacity:.95">
    Paste a Kirundi news article and the model tells you what it is about.
  </div>
  <div style="margin-top:14px;font-size:13px;opacity:.9">
    AfroXLMR fine-tuned on KIRNEWS &nbsp;·&nbsp; 12 topics &nbsp;·&nbsp; research demo
  </div>
</div>
"""

PLACEHOLDER = card(
    '<div style="text-align:center;color:#6b7280;padding:34px 8px">'
    '<div style="font-size:46px">📰</div>'
    '<div style="font-size:17px;margin-top:10px">Paste a Kirundi news article on the left<br>'
    'and press <b>Classify</b>.</div></div>')

FOOTER = ('<div style="text-align:center;color:#6b7280;font-size:13px;padding:10px 0">'
          'Built with Gradio &nbsp;·&nbsp; <a href="%s" target="_blank">Code &amp; report</a> '
          '&nbsp;·&nbsp; <a href="%s" target="_blank">Model</a></div>' % (GITHUB_URL, MODEL_URL))

ABOUT = f"""
### What this app does
It reads a Kirundi news text and predicts one of **12 topics**: politics, sport, economy, health,
entertainment, history, technology, culture, religion, environment, education, relationship.

### How it works
1. The text is split into sub-word tokens (only the **first {MAX_LEN} tokens** are used).
2. **AfroXLMR-base**, a Transformer pre-trained on 17 African languages including Kirundi, turns the tokens into a vector.
3. A small classification layer converts that vector into 12 scores, and *softmax* turns them into probabilities.

### Training and results
- Data: **KIRNEWS** (Niyongabo et al., 2020), cleaned to 1,776 unique articles after removing duplicates and conflicting labels.
- Fine-tuned with a class-weighted loss to help rare topics.
- Test set (267 articles): **accuracy 0.906**, **macro-F1 0.778**.

### Please keep in mind
- Rare topics (**history, religion, culture**) have very few training articles, so those predictions are less reliable.
- Some dataset labels are debatable and many articles cover more than one topic, so the model can disagree with the dataset label.
- The model has only seen Burundian news. It may be unreliable for other kinds of text or other languages.
- This is a research and education demo, not a tool for important decisions.

[Code, report and results]({GITHUB_URL}) &nbsp;·&nbsp; [Model on Hugging Face]({MODEL_URL})
"""


def level(p: float):
    if p >= 0.80:
        return "High confidence", "#166534", "#dcfce7"
    if p >= 0.50:
        return "Medium confidence", "#92400e", "#fef3c7"
    return "Low confidence", "#991b1b", "#fee2e2"


def warning_card(message: str) -> str:
    return card('<div style="text-align:center;color:#92400e;padding:26px 8px;font-size:16px">'
                '⚠️ %s</div>' % html.escape(message), border="#fcd34d")


def result_card(scores: dict, truncated: bool) -> str:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, p = ranked[0]
    text, fg, bg = level(p)
    bars = ""
    for i, (lab, pr) in enumerate(ranked[:5]):
        width = max(1, round(pr * 100))
        colour = "#4f46e5" if i == 0 else "#a5b4fc"
        bars += (
            '<div style="margin:12px 0">'
            '<div style="display:flex;justify-content:space-between;font-size:15px;margin-bottom:4px">'
            '<span>%s %s</span><b>%d%%</b></div>'
            '<div style="background:#eef2ff;border-radius:999px;height:10px;overflow:hidden">'
            '<div style="width:%d%%;height:10px;background:%s;border-radius:999px"></div></div></div>'
            % (ICONS.get(lab, ""), html.escape(lab), round(pr * 100), width, colour))
    note = ""
    if p < 0.50:
        note += ('<div style="margin-top:12px;font-size:14px;color:#991b1b">The model is not confident: '
                 'the text may be off-topic, very short, or not Kirundi.</div>')
    if truncated:
        note += ('<div style="margin-top:8px;font-size:13px;color:#6b7280">Only the first %d tokens '
                 'of the text were used.</div>' % MAX_LEN)
    inner = (
        '<div style="font-size:13px;letter-spacing:1px;color:#6b7280;text-transform:uppercase">'
        'Predicted topic</div>'
        '<div style="font-size:34px;font-weight:800;margin:6px 0 10px 0">%s %s</div>'
        '<span style="background:%s;color:%s;border-radius:999px;padding:4px 12px;font-size:14px;'
        'font-weight:600">%s &nbsp;·&nbsp; %d%%</span>'
        '<div style="margin-top:18px;font-size:13px;letter-spacing:1px;color:#6b7280;'
        'text-transform:uppercase">Top 5 topics</div>%s%s'
        % (ICONS.get(top, ""), html.escape(top.capitalize()), bg, fg, text, round(p * 100), bars, note))
    return card(inner)


# ------------------------------------------------------------------ prediction
@gpu
def predict(text: str) -> str:
    text = (text or "").strip()
    if len(text.split()) < MIN_WORDS:
        return warning_card("Please paste a longer Kirundi text (at least a full sentence).")
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu()
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    truncated = len(tokenizer(text)["input_ids"]) > MAX_LEN
    return result_card(scores, truncated)


def reset():
    return "", PLACEHOLDER


# -------------------------------------------------------------------- interface
CSS = ".gradio-container {max-width: 1100px !important; margin: auto !important;}"
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")
if int(gr.__version__.split(".")[0]) >= 6:     # Gradio 6+: theme and css belong to launch()
    demo = gr.Blocks(title="Kirundi News Topic Classifier")
    LAUNCH_KWARGS = {"theme": THEME, "css": CSS}
else:                                          # Gradio 4 / 5: they belong to Blocks
    demo = gr.Blocks(title="Kirundi News Topic Classifier", theme=THEME, css=CSS)
    LAUNCH_KWARGS = {}

with demo:
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    box = gr.Textbox(lines=13, label="Kirundi news text",
                                     placeholder="Paste a Kirundi news article or paragraph here...")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        go_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=1):
                    out = gr.HTML(PLACEHOLDER)
            if EXAMPLES:
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=box)
        with gr.Tab("About the model"):
            gr.Markdown(ABOUT)
    gr.HTML(FOOTER)
    go_btn.click(predict, inputs=box, outputs=out)
    clear_btn.click(reset, inputs=None, outputs=[box, out])

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
