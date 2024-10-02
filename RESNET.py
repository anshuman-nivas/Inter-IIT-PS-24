import os
import numpy as np
import torch
import math
from torchvision import models, transforms
from PIL import Image

# Directory where the GIFs are stored
gif_dir = "downloaded_gifs"  # Adjust this path
# Directory to save output features
output_features_dir = "OutputFeatures"

# Create the output directory if it doesn't exist
if not os.path.exists(output_features_dir):
    os.makedirs(output_features_dir)
    print(f"Created output features directory: {output_features_dir}")

# Load the pre-trained ResNet model and remove the fully connected layer
resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(
    *list(resnet_model.children())[:-1])  # Remove the last fc layer
resnet_model.eval()  # Set the model to evaluation mode

# Define the target size for resizing frames for ResNet input
target_size_resnet = (224, 224)  # Resize to 224x224 pixels

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(target_size_resnet),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Positional Encoding Function


def add_positional_encoding(features, max_len=5000):
    d_model = features.shape[1]  # 2048 in this case (ResNet feature size)
    pe = np.zeros((len(features), d_model))
    position = np.arange(0, len(features)).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) *
                      (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    # Add positional encoding to the features
    features_with_pe = features + pe
    return features_with_pe

# Function to extract important frames from a GIF


def extract_important_frames(gif_path, num_frames=16):
    try:
        with Image.open(gif_path) as img:
            total_frames = img.n_frames  # Total number of frames in the GIF
            extracted_frames = []  # List to hold the frames
            # Calculate step size for extracting frames
            step_size = max(1, total_frames // num_frames)

            for i in range(num_frames):
                frame_index = i * step_size
                if frame_index < total_frames:
                    img.seek(frame_index)  # Move to the selected frame
                    img = img.convert("RGB")  # Ensure image is in RGB format
                    img_tensor = preprocess(img)  # Apply preprocessing
                    # Store the tensor of the resized frame
                    extracted_frames.append(img_tensor)
                else:
                    break  # No more frames available
            return extracted_frames
    except EOFError:
        print(f"Reached end of frames for {gif_path}.")
    except Exception as e:
        print(f"Error extracting frames from {gif_path}: {e}")
        return []

# Function to extract ResNet features from the frames of a GIF


def extract_resnet_features(gif_file):
    gif_path = os.path.join(gif_dir, gif_file)
    frames = extract_important_frames(gif_path)
    if frames:
        features = []
        for frame in frames:
            with torch.no_grad():  # Disable gradient calculation
                frame = frame.unsqueeze(0)  # Add batch dimension
                # Extract features from the model
                feature = resnet_model(frame)
                # Store features as numpy array (2048-dimensional)
                features.append(feature.squeeze(
                    0).squeeze(-1).squeeze(-1).numpy())

        # Convert list of features to numpy array
        features_np = np.array(features)

        # Add positional encoding to the features
        features_with_pe = add_positional_encoding(features_np)

        # Save features with positional encoding to a .npy file
        feature_filename = os.path.basename(
            gif_file).replace('.gif', '_features_with_pe.npy')
        np.save(os.path.join(output_features_dir,
                feature_filename), features_with_pe)
        print(
            f"Extracted and saved features with positional encoding for {gif_file}")


# Process each GIF file in the directory
gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
for gif_file in gif_files:
    extract_resnet_features(gif_file)

print("\nResNet feature extraction with positional encoding complete for all GIFs!")
