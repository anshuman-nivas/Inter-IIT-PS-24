import os
import pandas as pd

# Load the dataset
df = pd.read_csv('tgif-v1.0.tsv', header=None,
                 sep='\t', names=['urls', 'Answers'])

# Extract Answers
gif_descriptions = df['Answers'].tolist()

# Folder paths
downloaded_gifs_folder = 'downloaded_gifs'  # Folder where GIFs are stored
description_folder = 'gif_descriptions'  # Folder to store descriptions
os.makedirs(description_folder, exist_ok=True)

# Get list of downloaded GIF files (assuming filenames like gif_1.gif, gif_2.gif, etc.)
downloaded_gifs = os.listdir(downloaded_gifs_folder)

# Loop through the downloaded GIFs and save corresponding descriptions
for gif_file in downloaded_gifs:
    # Extract the GIF number from the filename (assuming format like gif_1.gif)
    gif_number = int(gif_file.split('_')[1].split('.')[0])

    # Find the corresponding description (Answers column) based on the GIF number
    # Adjust to zero-based index
    description = gif_descriptions[gif_number - 1]

    # Save the description in a text file
    description_file = os.path.join(
        description_folder, f"description_{gif_number}.txt")
    with open(description_file, 'w') as desc_file:
        desc_file.write(description)

print("Descriptions for the downloaded GIFs have been saved.")
