import io
import os
import zipfile
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
TARGET_DIR = os.path.join("data", "raw")

def download_and_extract(url: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    print("Connecting to UCI Repository...")
    
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            zip_content = response.read()
            print("Download completed! Extracting archive...")
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
                zip_ref.extractall(output_path)
                
        print(f"Success! Files saved inside: {output_path}")
        print("Files present:")
        for file in os.listdir(output_path):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_and_extract(DATA_URL, TARGET_DIR)