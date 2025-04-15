'''
run this on a local machine, following the instructions below: 
1. Create a project on Google Cloud Console
3. Create a oAuth 2.0 client ID
4. Download the client_secret.json file and place it in the same folder as this script
5. Run this script, it will open a browser window for you to login to your google account and authorize the app
6. After you authorize the app, it will create a token.json file in the same folder as this script
7. Copy the token.json file to the server where you want to run the download_from_drive.py script
'''

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_FILE = "client_secret.json"  # Your downloaded file
# Authenticate using OAuth 2.0
flow = InstalledAppFlow.from_client_secrets_file(
    CLIENT_SECRET_FILE, SCOPES
)
creds = flow.run_local_server(port=0)

TOKEN_FILE = "token.json"  # File to store the access token
with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())

# Connect to Google Drive API
service = build("drive", "v3", credentials=creds)

print("authentication successful, created service")
print(service)