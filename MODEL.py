import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer
from PIL import Image
import imageio
from tqdm import tqdm  # For progress bar
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Pretrained ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Set ResNet to evaluation mode
# Remove classification layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Define transformation for images (ResNet expects 224x224 input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset for GIF Features and Descriptions


class GIFDataset(Dataset):
    def __init__(self, gif_dir, descriptions_dir, tokenizer, max_len=128):
        self.gif_dir = gif_dir
        self.descriptions_dir = descriptions_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]

    def load_description(self, gif_number):
        description_file = os.path.join(
            self.descriptions_dir, f"description_{gif_number}.txt")
        with open(description_file, "r") as file:
            description = file.read().strip()
        return description

    def extract_frames(self, gif_path, num_frames=16):
        reader = imageio.get_reader(gif_path)
        total_frames = reader.get_length()
        # Calculate step to get 16 frames
        step = max(1, total_frames // num_frames)
        frames = []
        for i in range(0, total_frames, step)[:num_frames]:
            frame = reader.get_data(i)
            pil_image = Image.fromarray(frame)

            # Ensure the image is in RGB mode (Convert P, L, RGBA to RGB)
            if pil_image.mode not in ['RGB']:
                pil_image = pil_image.convert('RGB')

            frames.append(pil_image)
        return frames

    def extract_resnet_features(self, frames):
        features = []
        for frame in frames:
            frame_tensor = transform(frame).unsqueeze(0)  # Prepare for ResNet
            with torch.no_grad():
                feature = resnet(frame_tensor).squeeze()  # Extract feature
            features.append(feature)
        return torch.stack(features)  # Shape: (16, 2048)

    def __len__(self):
        return len(self.gif_files)

    def __getitem__(self, idx):
        gif_number = int(self.gif_files[idx].split('_')[1].split('.')[0])

        # Load GIF and extract ResNet features
        gif_path = os.path.join(self.gif_dir, self.gif_files[idx])
        frames = self.extract_frames(gif_path)
        features = self.extract_resnet_features(frames)  # Shape: (16, 2048)

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

# RNN-based model for description generation


class GIFDescriptionRNN(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=30522, hidden_dim=512, num_layers=1):
        super(GIFDescriptionRNN, self).__init__()
        self.rnn = nn.RNN(input_size=feature_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.embedding = nn.Embedding(num_classes, hidden_dim)

    def forward(self, features, input_ids, attention_mask):
        # hidden_state shape: (1, batch_size, hidden_dim)
        rnn_out, hidden_state = self.rnn(features)
        # Use the last hidden state from RNN
        rnn_last_hidden = hidden_state[-1]

        # Embedding for text input
        # embedded_inputs shape: (batch_size, seq_len, hidden_dim)
        embedded_inputs = self.embedding(input_ids)

        # Combine the RNN output (visual features) with embedded text input
        combined = rnn_last_hidden.unsqueeze(
            1) + embedded_inputs  # Broadcasting hidden state

        # Predict next tokens in sequence (logits for token prediction)
        logits = self.fc_out(combined)
        return logits


# Load pre-trained tokenizer (BERT tokenizer for text processing)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare Dataset and DataLoader
gif_dir = "gifs2"  # Path to GIF files
descriptions_dir = "descriptions2"  # Path to text descriptions
dataset = GIFDataset(gif_dir, descriptions_dir, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, optimizer, and loss function
model = GIFDescriptionRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop with tqdm progress bar
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Progress bar for each epoch
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

        for batch in tepoch:
            features = batch['features']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, input_ids, attention_mask)

            # Compute loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update tqdm progress bar with the current loss
            tepoch.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

print("Training complete!")

# --- Testing the model on the test set ---


def test_model(test_gif_dir, test_descriptions_dir, model, tokenizer, batch_size=8):
    test_dataset = GIFDataset(test_gif_dir, test_descriptions_dir, tokenizer)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    generated_descriptions = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            features = batch['features']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model(features, input_ids, attention_mask)

            # Decode the output to generate predicted tokens
            predicted_tokens = torch.argmax(outputs, dim=-1)

            for i in range(len(predicted_tokens)):
                decoded_sentence = tokenizer.decode(
                    predicted_tokens[i], skip_special_tokens=True)
                generated_descriptions.append(decoded_sentence)

    return generated_descriptions


# Specify paths for the test set
test_gif_dir = r"D:\InterIIT\test_folder\test_gifs"  # Path to test GIFs
# Path to test descriptions
test_descriptions_dir = r"D:\InterIIT\test_folder\test_descriptions"

# Test the model and print out the generated descriptions
generated_descriptions = test_model(
    test_gif_dir, test_descriptions_dir, model, tokenizer)

for i, description in enumerate(generated_descriptions):
    print(f"Test GIF {i+1}: {description}")
