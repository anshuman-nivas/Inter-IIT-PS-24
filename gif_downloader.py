import requests
import os
import pandas as pd

# Load the dataset
df = pd.read_csv('tgif-v1.0.tsv', header=None,
                 sep='\t', names=['urls', 'Answers'])

# Extract URLs and Answers
X = df.drop('Answers', axis=1)
Y = df['Answers']

# Create a directory to store the downloaded GIFs
output_folder = 'downloaded_gifs'  # You can specify your preferred folder path
os.makedirs(output_folder, exist_ok=True)

# Extract URLs into a list
gif_urls = X['urls'].tolist()  # Use .tolist() to get the URLs as a list

# Function to download a GIF


def download_gif(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return True  # Indicate success
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False  # Indicate failure
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False  # Indicate failure

# Function to download GIFs based on a specific range


def download_gifs_in_range(start, end):
    successful_downloads = []  # List to keep track of successfully downloaded GIFs
    for index in range(start-1, end):  # Adjust to zero-based index
        url = gif_urls[index]
        gif_name = f"gif_{index + 1}.gif"
        save_path = os.path.join(output_folder, gif_name)
        print(f"Downloading {gif_name} from {url}")
        if download_gif(url, save_path):
            successful_downloads.append(url)  # Store successful download URL

    print("Download completed.")
    print(f"Successfully downloaded GIFs from: {successful_downloads}")


# Specify the range of GIFs to download
start_gif = 6007
end_gif = 10169

# Validate the input range
if 1 <= start_gif <= len(gif_urls) and 1 <= end_gif <= len(gif_urls) and start_gif <= end_gif:
    download_gifs_in_range(start_gif, end_gif)
else:
    print(f"Invalid range. Please enter a range between 1 and {
          len(gif_urls)}.")
