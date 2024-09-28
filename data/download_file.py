import os
import requests
from tqdm import tqdm

# Function to download the file with progress bar
def download_file(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    file_path = os.path.join(dest_folder, filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"{filename} already exists in {dest_folder}. Skipping download.")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(file_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"{filename} has been downloaded successfully.")

def main():
    # URL of the data to be downloaded (replace this with your actual data URL)
    data_urls = {
        'economic_data.csv': 'https://github.com/selva86/datasets/blob/master/economics.csv',
        'ecotourism_data.csv': 'https://github.com/selva86/datasets/blob/master/ecotourism_data.csv'
    }

    # Destination folder for downloaded data
    dest_folder = 'data'

    # Download each file
    for filename, url in data_urls.items():
        download_file(url, dest_folder, filename)

if __name__ == '__main__':
    main()
