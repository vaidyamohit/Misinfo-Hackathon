import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import nltk

# ✅ Fix Torch async issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ✅ Load Dataset
st.title("📰 TrueTell: AI-Powered Misinformation Detector")
st.write("🔍 Enter a statement below to check its credibility.")

dataset_path = "Dataset.xlsx"  # Change if using Google Sheets

@st.cache_data
def load_data():
    try:
        df_train = pd.read_excel(dataset_path, sheet_name="train")
        df_test = pd.read_excel(dataset_path, sheet_name="test")
        df_valid = pd.read_excel(dataset_path, sheet_name="valid")

        # ✅ Normalize column names (remove spaces, lowercase)
        df_train.columns = df_train.columns.str.strip().str.lower()
        df_test.columns = df_test.columns.str.strip().str.lower()
        df_valid.columns = df_valid.columns.str.strip().str.lower()

        return df_train, df_test, df_valid
    except FileNotFoundError:
        st.error("⚠️ Dataset.xlsx not found! Please upload the dataset.")
        return None, None, None

df_train, df_test, df_valid = load_data()

# ✅ Debug: Show available columns
if df_train is not None:
    st.write("Dataset Columns:", df_train.columns.tolist())

# ✅ Ensure "statement" column exists
if df_train is not None and "statement" not in df_train.columns:
    st.error("⚠️ Column 'statement' not found in dataset! Please check Dataset.xlsx")
    st.stop()

# ✅ Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df_train["clean_statement"] = df_train["statement"].apply(clean_text)
df_test["clean_statement"] = df_test["statement"].apply(clean_text)
df_valid["clean_statement"] = df_valid["statement"].apply(clean_text)

# ✅ Encode Labels
label_encoder = LabelEncoder()
df_train["label"] = label_encoder.fit_transform(df_train["label"])
df_test["label"] = label_encoder.transform(df_test["label"])
df_valid["label"] = label_encoder.transform(df_valid["label"])

# ✅ Train/Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["clean_statement"], df_train["label"], test_size=0.1, stratify=df_train["label"], random_state=42
)

# ✅ Train Logistic Regression (TF-IDF)
st.write("🚀 Training Logistic Regression Model...")
vectorizer = TfidfVectorizer(max_features=5000)
lr_clf = make_pipeline(vectorizer, LogisticRegression(max_iter=500))
lr_clf.fit(train_texts, train_labels)

# ✅ Define BERT+LSTM Model
class HybridBERTLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=6):
        super(HybridBERTLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])

# ✅ Train BERT+LSTM Model
st.write("🚀 Training BERT+LSTM Model (may take time)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_lstm_model = HybridBERTLSTM().to(device)
optimizer = torch.optim.AdamW(bert_lstm_model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Convert Data for BERT
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train BERT+LSTM
for epoch in range(1):  # 1 epoch to avoid long runtime
    bert_lstm_model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = bert_lstm_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

st.write("✅ Model Training Complete!")

# ✅ Streamlit UI for Predictions
model_choice = st.selectbox("Choose a model:", ["BERT+LSTM", "TF-IDF + Logistic Regression"])
user_text = st.text_area("📝 Enter a statement to analyze:")

if st.button("🔍 Analyze"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter a valid statement.")
    else:
        if model_choice == "BERT+LSTM":
            inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                output = bert_lstm_model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            pred_label = torch.argmax(output, dim=1).item()
        else:
            pred_label = lr_clf.predict([user_text])[0]

        # ✅ Display Prediction
        labels = ["False", "Half-True", "Mostly-True", "True", "Barely-True", "Pants-on-Fire"]
        st.success(f"✅ **Predicted Verdict:** {labels[pred_label]}")

st.markdown("---")
st.markdown("🔬 Developed with AI & Machine Learning | **TrueTell**")
