import os
import gdown

# Set the Google Drive file ID
file_id = '1_lNFU2oRJi-Ap42XviEppb0Do7ii9DIg'

# Set the destination directory and file name
destination_dir = 'data'
destination_file = os.path.join(destination_dir, 'prepared_results_simple.pt')

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Download the file using gdown
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output=destination_file, quiet=False)
print('File downloaded successfully.')
