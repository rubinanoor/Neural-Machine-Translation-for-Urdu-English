import os
import zipfile
import requests

def download_file(url, dest):
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# 1. Setup paths (Matching your --base-dir ./my_data_test)
raw_path = "./my_data_test/raw"
os.makedirs(raw_path, exist_ok=True)

# 2. Direct links to English-Urdu Moses files
links = {
    "TED2020": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-ur.txt.zip",
    "Tanzil": "https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/en-ur.txt.zip"
}

for name, url in links.items():
    zip_p = os.path.join(raw_path, f"{name}.zip")
    download_file(url, zip_p)
    
    with zipfile.ZipFile(zip_p, 'r') as zip_ref:
        zip_ref.extractall(raw_path)
    
    # Rename files to match your script's expected format: {Corpus}_{lang}.txt
    # The files inside the zip are usually named like 'TED2020.en-ur.en'
    os.rename(os.path.join(raw_path, f"{name}.en-ur.ur"), os.path.join(raw_path, f"{name}_ur.txt"))
    os.rename(os.path.join(raw_path, f"{name}.en-ur.en"), os.path.join(raw_path, f"{name}_en.txt"))
    print(f"Successfully prepared {name}")