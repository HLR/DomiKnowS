import requests

def download_file_from_gdrive(file_id, destination_file):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination_file)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination_file):
    CHUNK_SIZE = 32768
    with open(destination_file, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Set the Google Drive file ID
file_id = "1MUHUl0lmcVr4usXrZ_fm-YEbj0TfRoLJ"

# Set the destination file name
destination_file = "data/prepared_results.pt"

# Download the file
download_file_from_gdrive(file_id, destination_file)
print("File downloaded successfully.")
