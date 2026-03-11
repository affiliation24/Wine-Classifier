import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DOMAIN_WORDS = ['malbec','tannins','oak','champagne','citrus','sauvignon','pinot','merlot','barrique','fermentation']
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
NUM_CLASSES = 3
label_map = {0: 'Red', 1: 'White', 2: 'Rosé'}

def extract_domain_features(text):
    text_lower = text.lower()
    return [1.0 if word in text_lower else 0.0 for word in DOMAIN_WORDS]

class BertWineClassifier(nn.Module):
    def __init__(self, num_classes, domain_feature_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + domain_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, domain_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_embedding, domain_features], dim=1)
        return self.classifier(combined)

device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('wine_model/tokenizer')
model = BertWineClassifier(num_classes=NUM_CLASSES, domain_feature_dim=len(DOMAIN_WORDS))
model.load_state_dict(torch.load('wine_model/model.pt', map_location=device))
model.eval()

def respond(
    message,
    history: list[dict[str, str]],
):
    if len(message.strip()) < 10:
        yield "Please describe the wine in more detail so I can classify it! 🍷"
        return

    encoding = tokenizer(message, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    domain = torch.tensor([extract_domain_features(message)], dtype=torch.float32)

    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask'], domain)
        probs = torch.softmax(output, dim=1)[0]
        pred = output.argmax(dim=1).item()

    response  = f"**{label_map[pred]}**\n\n"
    response += f"Red:   {probs[0]:.1%}\n"
    response += f"White: {probs[1]:.1%}\n"
    response += f"Rosé:  {probs[2]:.1%}\n\n"
    response += "_Describe another wine to classify it!_"

    yield response

chatbot = gr.ChatInterface(
    respond,
    title="🍷 Wine Type Classifier",
    description="Describe a wine and I'll tell you if it's Red, White, or Rosé!",
    examples=[
        "Bold and fruity with hints of blackberry and vanilla oak",
        "Crisp and refreshing with citrus and green apple aromas",
        "Delicate and fresh with notes of strawberry and rose petals"
    ]
)

with gr.Blocks() as demo:
    chatbot.render()

if __name__ == "__main__":
    demo.launch()