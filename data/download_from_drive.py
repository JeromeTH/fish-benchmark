import requests
import os
import subprocess
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
import io
from google.oauth2.credentials import Credentials
import subprocess
from tqdm import tqdm
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "./creds/token.json"  # Create token file using interactive login on local machine
creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
service = build("drive", "v3", credentials=creds)

FOLDER_ID = "12dPNkATHe4xJP0DOFvI6ke6LoNT9KhIR"
def download_directory(folder_id, output_path):
    '''
    downloads all files in a folder to local using the same folder name. For all subfolders in it, recursively download them.
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, 
                               includeItemsFromAllDrives=True,  # Include shared files
                                supportsAllDrives=True,         # Include shared drive files
                                corpora="user").execute()
    items = results.get("files", [])
    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']
        if mime_type == 'application/vnd.google-apps.folder':
            new_output_path = os.path.join(output_path, file_name)
            download_directory(file_id, new_output_path)
        else:
            download_file(file_id, file_name, output_path)

def download_file(file_id, file_name, output_path):
    print(f"Downloading {file_name} from Google Drive to {output_path}...")
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(output_path, file_name)
    #create the file if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with io.FileIO(file_path, 'wb') as local_file:
        downloader = MediaIoBaseDownload(local_file, request)
        #use tqdm to show the progress bar
        done = False
        with tqdm(total=100, desc=f"Downloading {file_name}", unit="%", ncols=80) as pbar:
            while not done:
                status, done = downloader.next_chunk()
                pbar.update(int(status.progress() * 100) - pbar.n)  # Update progress bar
#download_file("1mobi4V34o39VuH6C5YuS2sF4HSWba6Paeznl7j-1HiI", "bites_training_data.csv", "./fish-benchmark/data")
#download_file('1PNXU9GO-6GCfWU6FMlThLfj1UR3-vO7c', "video.mp4", "./data")



def export_file(file_id, file_name, mime_type):
    request = service.files().export_media(fileId=file_id, mimeType=mime_type)
    # Specify the local file path to save the downloaded spreadsheet
    with io.FileIO(file_name, 'wb') as local_file:
        downloader = MediaIoBaseDownload(local_file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

if __name__ == "__main__":
    download_directory(FOLDER_ID, "/share/j_sun/jth264/bites_training_data")

'''

{
    'kind': 'drive#fileList',
    'incompleteSearch': False,
    'files': [
        {
            'kind': 'drive#file',
            'driveId': '0AOuhXaQjJM1cUk9PVA',
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'id': '1mobi4V34o39VuH6C5YuS2sF4HSWba6Paeznl7j-1HiI',
            'name': 'bites_training_data',
            'teamDriveId': '0AOuhXaQjJM1cUk9PVA'
        },
        {
            'kind': 'drive#file',
            'driveId': '0AOuhXaQjJM1cUk9PVA',
            'mimeType': 'application/vnd.google-apps.folder',
            'id': '12dPNkATHe4xJP0DOFvI6ke6LoNT9KhIR',
            'name': 'data',
            'teamDriveId': '0AOuhXaQjJM1cUk9PVA'
        }
    ]
}
# download_url = f"https://drive.google.com/uc?export=download&id=1mobi4V34o39VuH6C5YuS2sF4HSWba6Paeznl7j-1HiI"
# subprocess.run(["wget", download_url])


# video_url = "https://drive.google.com/file/d/1PNXU9GO-6GCfWU6FMlThLfj1UR3-vO7c/view?usp=drive_link"
# file_id = '1PNXU9GO-6GCfWU6FMlThLfj1UR3-vO7c'


'''