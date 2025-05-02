import os
import json
from boxsdk import Client
from boxsdk.auth.jwt_auth import JWTAuth
from tqdm import tqdm

# Load config.json
CONFIG_PATH = 'data/abby/config.json'
DESTINATION = '/share/j_sun/jth264/box_folder'
SHARED_LINK = 'https://cornell.box.com/s/cgj2d5vfiao87wwxv9h671sk4vwcmwy6'

# Set up Box JWT Auth
with open(CONFIG_PATH) as f:
    config = json.load(f)

auth = JWTAuth.from_settings_file(CONFIG_PATH)
client = Client(auth)
print("Box client initialized.")
# Resolve the shared folder link
shared_item = client.get_shared_item(SHARED_LINK)
if __name__ == "__main__":
    if shared_item.type != 'folder':
        raise ValueError("The shared link does not point to a folder.")
    print(f"Starting recursive download of Box folder: {shared_item.name}")
