import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Custom Dataset for GIF Features and Descriptions


class GIFDataset(Dataset):
    def __init__(self, features_dir, descriptions_dir, tokenizer, max_len=128):
        self.features_dir = features_dir
        self.descriptions_dir = descriptions_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_files = [f for f in os.listdir(
            features_dir) if f.endswith('_features_with_pe.npy')]

    def load_description(self, gif_number):
        # Load the description from the text file
        description_file = os.path.join(
            self.descriptions_dir, f"description_{gif_number}.txt")
        with open(description_file, "r") as file:
            description = file.read().strip()
        return description

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        # Extract the GIF number from the filename (assuming format like gif_1_features_with_pe.npy)
        gif_number = int(self.feature_files[idx].split('_')[1])

        # Load features
        feature_path = os.path.join(self.features_dir, self.feature_files[idx])
        features = np.load(feature_path)

        # Load the corresponding description
        description = self.load_description(gif_number)

        # Tokenize the description
        inputs = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

# Simple RNN-based model


class GIFDescriptionRNN(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=30522, hidden_dim=512, num_layers=1):
        super(GIFDescriptionRNN, self).__init__()
        self.rnn = nn.RNN(input_size=feature_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.embedding = nn.Embedding(num_classes, hidden_dim)

    def forward(self, features, input_ids, attention_mask):
        # RNN for GIF features
        rnn_out, _ = self.rnn(features)

        # Embedding layer for text input (input_ids)
        embedded_inputs = self.embedding(input_ids)

        # Combine the RNN output with the embedded input
        # (For simplicity, we could concatenate or use the last hidden state from the RNN)
        combined = rnn_out + embedded_inputs  # Basic combination

        # Predict next tokens in sequence
        logits = self.fc_out(combined)
        return logits


# Load pre-trained tokenizer (BERT tokenizer for text processing)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare Dataset and DataLoader
features_dir = "OutputFeatures"  # Path to features with positional encoding
descriptions_dir = "gif_descriptions"  # Path to text descriptions
dataset = GIFDataset(features_dir, descriptions_dir, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, optimizer, and loss function
model = GIFDescriptionRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        features = batch['features']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         input_ids.view(-1))
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

print("Training complete!")
