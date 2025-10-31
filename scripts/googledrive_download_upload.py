import os
import csv
import sys
import json
import hashlib
import logging
import re
import cloudinary.uploader
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import tempfile
from datetime import datetime
import pickle

# Setup directory structure
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import USE_FILENAME, UNIQUE_FILENAME
except ImportError:
    # If config import fails, use default values
    USE_FILENAME = True
    UNIQUE_FILENAME = False

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Google Drive API scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.file'  # Needed for permission management
]

# Configure logging
def setup_logging(folder_name):
    """Setup logging with timestamp and folder-specific log file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'{folder_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_filename

def extract_folder_id_from_url(url_or_id):
    """
    Extract Google Drive folder ID from various URL formats or return the ID if already provided.
    
    Supported formats:
    - https://drive.google.com/drive/folders/FOLDER_ID
    - https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
    - https://drive.google.com/drive/u/0/folders/FOLDER_ID
    - https://drive.google.com/folderview?id=FOLDER_ID
    - FOLDER_ID (direct ID)
    
    Args:
        url_or_id (str): Google Drive URL or folder ID
        
    Returns:
        str: Extracted folder ID or None if invalid
    """
    if not url_or_id:
        return None
    
    # If it's already just an ID (no slashes or protocols), return it
    if not ('/' in url_or_id or 'http' in url_or_id.lower()):
        return url_or_id.strip()
    
    # Pattern to match various Google Drive folder URL formats
    patterns = [
        r'drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/folderview\?id=([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    # If no pattern matches, try to extract any ID-like string
    id_pattern = r'([a-zA-Z0-9_-]{25,})'
    match = re.search(id_pattern, url_or_id)
    if match:
        return match.group(1)
    
    return None

def validate_folder_id(service, folder_id):
    """
    Validate that the folder ID exists and is accessible.
    
    Args:
        service: Google Drive service instance
        folder_id (str): Google Drive folder ID
        
    Returns:
        tuple: (is_valid, folder_name, error_message)
    """
    try:
        # First, try to get basic folder info
        folder_info = service.files().get(
            fileId=folder_id, 
            fields="name, mimeType, capabilities, owners, shared, parents",
            supportsAllDrives=True  # Support Shared Drives
        ).execute()
        
        print(f"🔍 Folder info retrieved:")
        print(f"  Name: {folder_info.get('name', 'Unknown')}")
        print(f"  MIME Type: {folder_info.get('mimeType', 'Unknown')}")
        print(f"  Shared: {folder_info.get('shared', False)}")
        
        # Check if it's actually a folder
        if folder_info.get('mimeType') != 'application/vnd.google-apps.folder':
            return False, None, f"ID '{folder_id}' is not a folder"
        
        folder_name = folder_info.get('name', 'Unknown')
        
        # Check capabilities
        capabilities = folder_info.get('capabilities', {})
        can_list_children = capabilities.get('canListChildren', False)
        
        print(f"🔐 Permissions:")
        print(f"  Can list children: {can_list_children}")
        print(f"  Can read: {capabilities.get('canRead', False)}")
        print(f"  Can edit: {capabilities.get('canEdit', False)}")
        
        if not can_list_children:
            # Try a test query to see if we can actually list files
            try:
                test_query = f"'{folder_id}' in parents"
                test_result = service.files().list(
                    q=test_query, 
                    pageSize=1, 
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                print(f"📋 Test query successful - folder is accessible")
                return True, folder_name, None
                
            except Exception as test_error:
                print(f"❌ Test query failed: {str(test_error)}")
                return False, folder_name, f"Cannot access folder contents: {str(test_error)}"
        
        return True, folder_name, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error details: {error_msg}")
        
        if "notFound" in error_msg:
            return False, None, f"Folder not found: {folder_id}"
        elif "forbidden" in error_msg.lower() or "insufficientPermissions" in error_msg:
            return False, None, f"Insufficient permissions to access folder: {folder_id}"
        else:
            return False, None, f"Error accessing folder: {error_msg}"

# Thread-safe counter and lock for progress tracking
progress_lock = Lock()
progress_counter = {'uploaded': 0, 'failed': 0, 'total': 0, 'skipped': 0}
error_log = []

class UploadCache:
    """Manages the cache of uploaded files to support resume functionality"""
    
    def __init__(self, folder_path: str):
        """Initialize cache for a specific Google Drive folder"""
        self.folder_path = folder_path
        
        # Create a unique cache file name based on the folder path
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()
        self.cache_file = os.path.join(CACHE_DIR, f'gdrive_upload_cache_{folder_hash}.json')
        self.lock = Lock()
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load the cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
        return {
            'folder_path': '',
            'last_run': '',
            'successful_uploads': {},
            'failed_uploads': {}
        }
    
    def _save_cache(self):
        """Save the cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
    
    def is_uploaded(self, file_id: str) -> bool:
        """Check if a file was successfully uploaded in previous runs"""
        with self.lock:
            return file_id in self.cache['successful_uploads']
    
    def mark_uploaded(self, file_id: str, result: dict):
        """Mark a file as successfully uploaded"""
        with self.lock:
            self.cache['folder_path'] = self.folder_path
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['successful_uploads'][file_id] = {
                'timestamp': datetime.now().isoformat(),
                'cloudinary_url': result['cloudinary_url'],
                'public_id': result.get('public_id', ''),
                'filename': result.get('filename', '')
            }
            if file_id in self.cache['failed_uploads']:
                del self.cache['failed_uploads'][file_id]
            self._save_cache()
    
    def mark_failed(self, file_id: str, error: str):
        """Mark a file as failed upload"""
        with self.lock:
            self.cache['folder_path'] = self.folder_path
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['failed_uploads'][file_id] = {
                'timestamp': datetime.now().isoformat(),
                'error': error
            }
            self._save_cache()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'successful': len(self.cache['successful_uploads']),
            'failed': len(self.cache['failed_uploads']),
            'last_run': self.cache['last_run']
        }

def authenticate_google_drive():
    """Authenticate and return Google Drive service"""
    creds = None
    token_file = os.path.join(CACHE_DIR, 'token.pickle')
    
    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            credentials_file = 'credentials.json'  # Download from Google Cloud Console
            if not os.path.exists(credentials_file):
                print("Error: credentials.json not found!")
                print("Please download credentials.json from Google Cloud Console:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a project or select existing one")
                print("3. Enable Google Drive API")
                print("4. Create credentials (OAuth 2.0 Client ID)")
                print("5. Download credentials.json and place it in the root directory")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def get_google_drive_download_url(service, file_id):
    """
    Get the proper download URL for a Google Drive file, handling shared drives.
    """
    try:
        # Get file metadata with additional fields for shared drives
        file = service.files().get(
            fileId=file_id, 
            fields='webContentLink,exportLinks,driveId,parents,mimeType',
            supportsAllDrives=True
        ).execute()
        
        # For regular files, use webContentLink if available
        if 'webContentLink' in file:
            return file['webContentLink']
        
        # For Google Docs, Sheets, etc., use export links
        if 'exportLinks' in file:
            # For images, we want the original format
            for format_type in ['image/jpeg', 'image/png', 'application/pdf']:
                if format_type in file['exportLinks']:
                    return file['exportLinks'][format_type]
        
        # For shared drive files or when webContentLink is not available
        # Use the Google Drive API v3 direct download endpoint
        return f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        
    except Exception as e:
        logging.error(f"Error getting download URL for {file_id}: {e}")
        # Fallback: construct traditional download URL
        return f"https://drive.google.com/uc?id={file_id}&export=download"

def upload_single_image_from_gdrive(service, file_info, base_folder_name, cache):
    """
    Download file from Google Drive to a temporary location, then upload to Cloudinary.
    """
    file_id = file_info['id']
    filename = file_info['name']
    folder_path = file_info.get('folder_path', '')
    
    # Create the full Cloudinary folder path
    if folder_path:
        cloudinary_folder = f"{base_folder_name}/{folder_path}"
    else:
        cloudinary_folder = base_folder_name
    
    logging.info(f"Processing: {filename} (ID: {file_id}) → {cloudinary_folder}")
    
    # Already uploaded? return cached result and update progress
    if cache.is_uploaded(file_id):
        with progress_lock:
            progress_counter['skipped'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SKIPPED: {filename} (previously uploaded)")
            
            # Real-time progress logging every 10 files or at completion
            if current % 10 == 0 or current == progress_counter['total']:
                progress_msg = (f"PROGRESS UPDATE: {current}/{progress_counter['total']} "
                               f"(✓Success: {progress_counter['uploaded']}, ❌Failed: {progress_counter['failed']}, "
                               f"⏭️Skipped: {progress_counter['skipped']})")
                logging.info(progress_msg)
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        cached_data = cache.cache['successful_uploads'][file_id]
        return {
            'local_filename': os.path.splitext(filename)[0],
            'cloudinary_url': cached_data['cloudinary_url'],
            'status': 'skipped',
            'public_id': cached_data.get('public_id', ''),
            'folder_path': folder_path
        }

    temp_file_path = None
    
    try:
        logging.info(f"START TRANSFER: {filename} → {cloudinary_folder}")
        
        # 1) Download file from Google Drive to temporary location
        logging.info(f"  Downloading from Google Drive: {filename}")
        
        # Get file metadata first
        try:
            file_metadata = service.files().get(
                fileId=file_id, 
                fields='name,mimeType,size',
                supportsAllDrives=True
            ).execute()
            
            file_size = int(file_metadata.get('size', 0))
            logging.info(f"  File size: {file_size} bytes")
            
        except Exception as e:
            logging.warning(f"  Could not get file metadata: {e}")
        
        # Download the file content
        request = service.files().get_media(fileId=file_id)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            
            # Download in chunks for better memory management
            downloader = MediaIoBaseDownload(temp_file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logging.info(f"  Download progress: {int(status.progress() * 100)}%")
        
        logging.info(f"  Downloaded to temporary file: {temp_file_path}")

        # 2) Derive filename / extension for Cloudinary options
        file_stem, ext = os.path.splitext(filename)
        # Clean the file stem to avoid whitespace issues
        file_stem = file_stem.strip()
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        logging.info(f"  Uploading to Cloudinary as: {file_stem}.{original_extension}")

        # 3) Upload the temporary file to Cloudinary
        response = cloudinary.uploader.upload(
            temp_file_path,
            folder=cloudinary_folder,       # Use the organized folder path
            public_id=file_stem,            # use original filename without extension
            use_filename=False,             # don't use local filename
            unique_filename=False,          # keep stable public_id
            overwrite=True,                 # allow re-runs to overwrite
            format=original_extension,      # keep original extension
            resource_type="image"           # or "auto" if you might have videos/svg/others
        )

        result = {
            'local_filename': file_stem,
            'cloudinary_url': response['secure_url'],
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'filename': filename,
            'folder_path': folder_path
        }

        # 4) Cache + progress + real-time logging
        cache.mark_uploaded(file_id, result)
        with progress_lock:
            progress_counter['uploaded'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SUCCESS: {filename} → {result['cloudinary_url']}")
            
            # Real-time progress logging every 10 files or at completion
            if current % 10 == 0 or current == progress_counter['total']:
                progress_msg = (f"PROGRESS UPDATE: {current}/{progress_counter['total']} "
                               f"(✓Success: {progress_counter['uploaded']}, ❌Failed: {progress_counter['failed']}, "
                               f"⏭️Skipped: {progress_counter['skipped']})")
                logging.info(progress_msg)
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        return result

    except Exception as e:
        error_message = str(e)
        with progress_lock:
            progress_counter['failed'] += 1
            if len(error_log) < 10:
                error_log.append(f"{filename}: {error_message}")
            logging.error(f"FAILED: {filename}")
            logging.error(f"Error details: {error_message}")
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            
            # Real-time progress logging every 10 files or at completion
            if current % 10 == 0 or current == progress_counter['total']:
                progress_msg = (f"PROGRESS UPDATE: {current}/{progress_counter['total']} "
                               f"(✓Success: {progress_counter['uploaded']}, ❌Failed: {progress_counter['failed']}, "
                               f"⏭️Skipped: {progress_counter['skipped']})")
                logging.info(progress_msg)
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        file_stem = os.path.splitext(filename)[0]
        cache.mark_failed(file_id, error_message)
        
        return {
            'local_filename': file_stem,
            'cloudinary_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message,
            'folder_path': folder_path
        }
    
    finally:
        # 5) Cleanup: Remove temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logging.info(f"  Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logging.warning(f"  Could not remove temporary file {temp_file_path}: {cleanup_error}")

def test_cloudinary_connection():
    """Test if Cloudinary is properly configured."""
    try:
        if not cloudinary.config().cloud_name:
            return False, "Cloudinary cloud_name is not configured"
        if not cloudinary.config().api_key:
            return False, "Cloudinary api_key is not configured"
        if not cloudinary.config().api_secret:
            return False, "Cloudinary api_secret is not configured"
        
        return True, "Cloudinary configuration looks good"
    except Exception as e:
        return False, f"Configuration error: {str(e)}"

def test_google_drive_connection():
    """Test if Google Drive is properly configured."""
    try:
        service = authenticate_google_drive()
        if not service:
            return False, "Google Drive authentication failed"
        
        # Test the connection by listing the root folder
        results = service.files().list(pageSize=1).execute()
        return True, "Google Drive connection successful"
    except Exception as e:
        return False, f"Google Drive connection error: {str(e)}"

def list_all_google_drive_folders(service, parent_id='root', indent=0, parent_path=''):
    """
    Recursively list all folders in Google Drive.
    
    Args:
        service: Google Drive service instance
        parent_id (str): Parent folder ID (use 'root' for root folder)
        indent (int): Indentation level for display
        parent_path (str): Path to parent folder for display
        
    Returns:
        list: List of folder information dictionaries
    """
    folders = []
    
    try:
        # Query for folders only
        query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        for item in items:
            folder_path = f"{parent_path}/{item['name']}" if parent_path else f"/{item['name']}"
            folder_info = {
                'id': item['id'],
                'name': item['name'],
                'path': folder_path
            }
            folders.append(folder_info)
            print(f"{'  ' * indent}📁 {folder_path} (ID: {item['id']})")
            
            # Recursively list subfolders
            subfolders = list_all_google_drive_folders(service, item['id'], indent + 1, folder_path)
            folders.extend(subfolders)
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{parent_id}': {str(e)}")
    
    return folders

def list_shared_drives(service):
    """
    List all Shared Drives (Team Drives) the user has access to.
    
    Args:
        service: Google Drive service instance
        
    Returns:
        list: List of shared drive information
    """
    shared_drives = []
    
    try:
        page_token = None
        
        while True:
            results = service.drives().list(
                pageToken=page_token,
                fields="nextPageToken, drives(id, name, capabilities)"
            ).execute()
            
            drives = results.get('drives', [])
            
            for drive in drives:
                drive_info = {
                    'id': drive['id'],
                    'name': drive['name'],
                    'capabilities': drive.get('capabilities', {})
                }
                shared_drives.append(drive_info)
                print(f"🚗 Shared Drive: {drive['name']} (ID: {drive['id']})")
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing shared drives: {str(e)}")
        # If drives API is not available, it might be due to permissions
        if "drives" in str(e).lower():
            print("Note: Shared Drives access might require additional permissions")
    
    return shared_drives

def list_folders_in_shared_drive(service, drive_id, drive_name):
    """
    List all folders in a specific Shared Drive.
    
    Args:
        service: Google Drive service instance
        drive_id (str): Shared Drive ID
        drive_name (str): Shared Drive name for display
        
    Returns:
        list: List of folder information
    """
    folders = []
    
    try:
        page_token = None
        
        while True:
            results = service.files().list(
                q=f"mimeType='application/vnd.google-apps.folder' and trashed=false",
                driveId=drive_id,
                corpora='drive',
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, parents)"
            ).execute()
            
            items = results.get('files', [])
            
            for item in items:
                folder_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'drive_name': drive_name,
                    'drive_id': drive_id,
                    'parents': item.get('parents', [])
                }
                folders.append(folder_info)
                print(f"  📁 {item['name']} (ID: {item['id']}) - in Shared Drive: {drive_name}")
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing folders in shared drive '{drive_name}': {str(e)}")
    
    return folders

def list_shared_with_me(service):
    """
    List all files and folders shared with me.
    
    Args:
        service: Google Drive service instance
        
    Returns:
        dict: Dictionary with shared folders and files
    """
    shared_items = {'folders': [], 'files': []}
    
    try:
        # Query for items shared with me
        query = "sharedWithMe=true and trashed=false"
        page_token = None
        
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, owners, shared, parents)",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True
            ).execute()
            
            items = results.get('files', [])
            
            for item in items:
                owner_info = item.get('owners', [{}])[0]
                owner_name = owner_info.get('displayName', 'Unknown')
                
                item_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'owner': owner_name,
                    'shared': item.get('shared', False),
                    'source': 'shared_with_me'
                }
                
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    shared_items['folders'].append(item_info)
                    print(f"📁 {item['name']} (ID: {item['id']}) - Owner: {owner_name}")
                else:
                    # Check if it's an image file
                    if item['mimeType'].startswith('image/'):
                        shared_items['files'].append(item_info)
                        print(f"🖼️  {item['name']} (ID: {item['id']}) - Owner: {owner_name}")
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing shared files: {str(e)}")
    
    return shared_items

def list_all_shared_content(service):
    """
    List all shared content including both files shared with me and Shared Drives.
    
    Args:
        service: Google Drive service instance
        
    Returns:
        dict: Dictionary with all shared content
    """
    all_shared = {'folders': [], 'files': [], 'shared_drives': []}
    
    print("📋 Scanning files and folders shared with you...")
    shared_items = list_shared_with_me(service)
    all_shared['folders'].extend(shared_items['folders'])
    all_shared['files'].extend(shared_items['files'])
    
    print(f"\n🚗 Scanning Shared Drives (Team Drives)...")
    shared_drives = list_shared_drives(service)
    all_shared['shared_drives'] = shared_drives
    
    # List folders in each Shared Drive
    shared_drive_folders = []
    for drive in shared_drives:
        print(f"\n  📁 Scanning folders in Shared Drive: {drive['name']}")
        folders = list_folders_in_shared_drive(service, drive['id'], drive['name'])
        for folder in folders:
            folder['source'] = 'shared_drive'
        shared_drive_folders.extend(folders)
    
    all_shared['folders'].extend(shared_drive_folders)
    
    return all_shared
    """
    List all files and folders shared with me.
    
    Args:
        service: Google Drive service instance
        
    Returns:
        dict: Dictionary with shared folders and files
    """
    shared_items = {'folders': [], 'files': []}
    
    try:
        # Query for items shared with me
        query = "sharedWithMe=true and trashed=false"
        page_token = None
        
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, owners, shared, parents)",
                pageToken=page_token
            ).execute()
            
            items = results.get('files', [])
            
            for item in items:
                owner_info = item.get('owners', [{}])[0]
                owner_name = owner_info.get('displayName', 'Unknown')
                
                item_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'owner': owner_name,
                    'shared': item.get('shared', False)
                }
                
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    shared_items['folders'].append(item_info)
                    print(f"📁 {item['name']} (ID: {item['id']}) - Owner: {owner_name}")
                else:
                    # Check if it's an image file
                    if item['mimeType'].startswith('image/'):
                        shared_items['files'].append(item_info)
                        print(f"🖼️  {item['name']} (ID: {item['id']}) - Owner: {owner_name}")
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing shared files: {str(e)}")
    
    return shared_items

def get_images_from_gdrive_folder(service, folder_id, recursive=True, parent_path='', folder_name=''):
    """
    Get list of image files from a Google Drive folder, optionally including subfolders.
    
    Args:
        service: Google Drive service instance
        folder_id (str): Folder ID in Google Drive
        recursive (bool): If True, scan subfolders recursively
        parent_path (str): Path to parent folder for organizing in Cloudinary
        folder_name (str): Current folder name for logging
        
    Returns:
        list: List of image file information dictionaries with folder_path for organization
    """
    image_mimetypes = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
        'image/bmp', 'image/webp', 'image/svg+xml'
    }
    image_files = []
    
    try:
        # Build query for image files in the current folder
        mimetype_query = " or ".join([f"mimeType='{mt}'" for mt in image_mimetypes])
        query = f"'{folder_id}' in parents and ({mimetype_query}) and trashed=false"
        
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            items = results.get('files', [])
            for item in items:
                image_files.append({
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'size': item.get('size', 0),
                    'folder_path': parent_path,  # Store the folder path for Cloudinary organization
                    'folder_name': folder_name
                })
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
        # If recursive, also scan subfolders
        if recursive:
            # Get all subfolders
            folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            page_token = None
            
            while True:
                results = service.files().list(
                    q=folder_query,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                subfolders = results.get('files', [])
                for subfolder in subfolders:
                    subfolder_path = f"{parent_path}/{subfolder['name']}" if parent_path else subfolder['name']
                    print(f"  📁 Scanning subfolder: {subfolder_path}")
                    
                    # Recursively get images from subfolder
                    subfolder_images = get_images_from_gdrive_folder(
                        service, 
                        subfolder['id'], 
                        recursive=True, 
                        parent_path=subfolder_path,
                        folder_name=subfolder['name']
                    )
                    image_files.extend(subfolder_images)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{folder_id}': {str(e)}")
    
    return image_files

def upload_gdrive_folder_to_cloudinary(folder_id, folder_name=None, max_workers=10, recursive=True):
    """
    Upload images from a Google Drive folder to Cloudinary using multi-threading.
    Supports resuming interrupted uploads through caching and recursive subfolder scanning.
    
    Args:
        folder_id (str): Google Drive folder ID
        folder_name (str): Optional custom folder name for Cloudinary (default: uses Drive folder name)
        max_workers (int): Number of concurrent upload threads (default: 10)
        recursive (bool): If True, scan and upload from subfolders recursively (default: True)
    """
    
    # Test connections
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print("\n⚠️  Please check your config.py and .env file")
        return
    
    print("  ✓ Cloudinary verified\n")
    
    print("Testing Google Drive connection...")
    is_connected, message = test_google_drive_connection()
    print(f"  {message}")
    
    if not is_connected:
        print("\n⚠️  Please check your Google Drive credentials")
        return
    
    print("  ✓ Google Drive verified\n")
    
    # Initialize Google Drive service
    service = authenticate_google_drive()
    if not service:
        print("Failed to authenticate Google Drive")
        return
    
    # Get folder name if not provided
    if not folder_name:
        try:
            folder_info = service.files().get(fileId=folder_id, fields="name").execute()
            folder_name = folder_info['name']
        except Exception as e:
            print(f"Error getting folder name: {e}")
            folder_name = f"gdrive_folder_{folder_id}"
    
    # Get all images from Google Drive folder (with recursive scanning)
    print(f"Scanning Google Drive folder: {folder_name} (ID: {folder_id})")
    if recursive:
        print("  📁 Recursive scanning enabled - will include subfolders")
    else:
        print("  📁 Scanning current folder only")
    
    image_files = get_images_from_gdrive_folder(service, folder_id, recursive=recursive, folder_name=folder_name)
    
    if not image_files:
        print(f"No images found in folder '{folder_name}'" + (" (including subfolders)" if recursive else ""))
        return
    
    # Count files by folder for summary
    folder_counts = {}
    for img in image_files:
        folder_path = img.get('folder_path', '')
        display_folder = folder_path if folder_path else '(root folder)'
        if display_folder not in folder_counts:
            folder_counts[display_folder] = 0
        folder_counts[display_folder] += 1
    
    print(f"\n📊 Found {len(image_files)} images across {len(folder_counts)} folder(s):")
    for folder_path, count in sorted(folder_counts.items()):
        print(f"  📁 {folder_path}: {count} images")
    print()
    
    # Setup logging
    log_file = setup_logging(folder_name)
    
    # Log the initial scan results
    logging.info(f"INITIAL FOLDER SCAN RESULTS:")
    logging.info(f"Found {len(image_files)} images across {len(folder_counts)} folder(s):")
    for folder_path, count in sorted(folder_counts.items()):
        cloudinary_path = f"{folder_name}/{folder_path}" if folder_path != '(root folder)' else folder_name
        logging.info(f"  📁 {folder_path}: {count} images → will be uploaded to: {cloudinary_path}")
    logging.info("")
    
    # Generate CSV filename based on folder name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(OUTPUT_DIR, f"{folder_name}_{timestamp}.csv")
    
    # Initialize upload cache
    cache = UploadCache(folder_id)
    cache_stats = cache.get_stats()
    
    # Initialize progress counter
    progress_counter['total'] = len(image_files)
    progress_counter['uploaded'] = 0
    progress_counter['failed'] = 0
    progress_counter['skipped'] = 0
    error_log.clear()
    
    logging.info(f"Processing folder: {folder_name}")
    logging.info(f"Recursive scanning: {recursive}")
    logging.info(f"Found {len(image_files)} images to process across {len(folder_counts)} folder(s)")
    if cache_stats['successful'] > 0:
        logging.info(f"Cache found: {cache_stats['successful']} previously uploaded files will be skipped")
        logging.info(f"Last upload run: {cache_stats['last_run']}")
    logging.info(f"Using {max_workers} concurrent threads for faster upload")
    logging.info(f"Output will be saved to: {output_csv}")
    logging.info(f"Log file: {log_file}\n")
    
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for concurrent uploads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks - pass folder_name as base folder
        future_to_image = {
            executor.submit(upload_single_image_from_gdrive, service, img, folder_name, cache): img 
            for img in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                img = future_to_image[future]
                print(f"❌ Unexpected error processing {img['name']}: {e}")
                results.append({
                    'local_filename': os.path.splitext(img['name'])[0],
                    'cloudinary_url': 'UPLOAD_FAILED',
                    'status': 'failed',
                    'error': str(e),
                    'folder_path': img.get('folder_path', '')
                })
    
    elapsed_time = time.time() - start_time
    
    # Calculate detailed statistics per folder
    stats_by_folder = {}
    stats_by_status = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for result in results:
        folder_path = result.get('folder_path', '')
        display_folder = folder_path if folder_path else '(root folder)'
        status = result.get('status', 'unknown')
        
        # Initialize folder stats if not exists
        if display_folder not in stats_by_folder:
            stats_by_folder[display_folder] = {
                'success': 0, 'failed': 0, 'skipped': 0, 'total': 0,
                'cloudinary_path': f"{folder_name}/{folder_path}" if folder_path else folder_name
            }
        
        # Update folder-specific stats
        if status in stats_by_folder[display_folder]:
            stats_by_folder[display_folder][status] += 1
        stats_by_folder[display_folder]['total'] += 1
        
        # Update overall stats
        if status in stats_by_status:
            stats_by_status[status] += 1
    
    # Write results to CSV
    if results:
        csv_columns = ['local_filename', 'cloudinary_url', 'folder_path']
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            
            successful = stats_by_status['success']
            failed = stats_by_status['failed']
            skipped = stats_by_status['skipped']
            
            # Log detailed statistics
            logging.info(f"\n{'='*60}")
            logging.info(f"UPLOAD OPERATION COMPLETED")
            logging.info(f"{'='*60}")
            logging.info(f"Operation completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                logging.info(f"Average upload speed: {successful/elapsed_time:.2f} images/second")
            logging.info(f"Results saved to: {output_csv}")
            logging.info("")
            
            # Overall statistics
            logging.info(f"OVERALL STATISTICS:")
            logging.info(f"  Total images processed: {len(image_files)}")
            logging.info(f"  Successfully uploaded: {successful}")
            logging.info(f"  Previously uploaded (skipped): {skipped}")
            logging.info(f"  Failed uploads: {failed}")
            logging.info(f"  Recursive scanning: {recursive}")
            logging.info(f"  Total folders processed: {len(stats_by_folder)}")
            logging.info("")
            
            # Detailed per-folder statistics
            logging.info(f"DETAILED FOLDER STATISTICS:")
            logging.info(f"{'Folder Path':<40} {'Success':<8} {'Failed':<8} {'Skipped':<8} {'Total':<8} {'Cloudinary Path'}")
            logging.info(f"{'-'*120}")
            
            for folder_path in sorted(stats_by_folder.keys()):
                stats = stats_by_folder[folder_path]
                logging.info(f"{folder_path:<40} {stats['success']:<8} {stats['failed']:<8} {stats['skipped']:<8} {stats['total']:<8} {stats['cloudinary_path']}")
            
            logging.info("")
            
            # Console output
            print(f"\n{'='*60}")
            print(f"✓ Upload completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                print(f"✓ Average speed: {successful/elapsed_time:.2f} images/second")
            print(f"✓ Results saved to '{output_csv}'")
            print(f"  Total images processed: {len(image_files)}")
            print(f"  Successfully uploaded: {successful}")
            print(f"  Previously uploaded (skipped): {skipped}")
            print(f"  Failed uploads: {failed}")
            
            # Show detailed folder organization summary
            if len(stats_by_folder) > 1:  # Only show if multiple folders
                print(f"\n📁 Detailed Upload Statistics by Folder:")
                print(f"{'Folder':<30} {'✓Success':<10} {'❌Failed':<10} {'⏭️Skipped':<10} {'📊Total':<10}")
                print(f"{'-'*70}")
                
                for folder_path in sorted(stats_by_folder.keys()):
                    stats = stats_by_folder[folder_path]
                    folder_display = folder_path[:27] + "..." if len(folder_path) > 30 else folder_path
                    print(f"{folder_display:<30} {stats['success']:<10} {stats['failed']:<10} {stats['skipped']:<10} {stats['total']:<10}")
                
                print(f"\n� Cloudinary Organization:")
                for folder_path in sorted(stats_by_folder.keys()):
                    stats = stats_by_folder[folder_path]
                    if stats['success'] > 0:  # Only show folders with successful uploads
                        print(f"  📂 {stats['cloudinary_path']}: {stats['success']} images uploaded")
            
            # Cache statistics
            cache_stats = cache.get_stats()
            print(f"\n💾 Cache Status:")
            print(f"  Total files in cache: {cache_stats['successful']}")
            print(f"  Failed files in cache: {cache_stats['failed']}")
            print(f"  Last upload run: {cache_stats['last_run']}")
            
            # Log cache statistics
            logging.info(f"CACHE STATISTICS:")
            logging.info(f"  Total files in cache: {cache_stats['successful']}")
            logging.info(f"  Failed files in cache: {cache_stats['failed']}")
            logging.info(f"  Last upload run: {cache_stats['last_run']}")
            logging.info("")
            
            # Sample errors
            if error_log:
                print(f"\n⚠️  Sample errors (first 10):")
                logging.info(f"SAMPLE ERRORS (first 10):")
                for err in error_log:
                    print(f"  - {err}")
                    logging.info(f"  - {err}")
                logging.info("")
            
            # Final summary in log
            logging.info(f"OPERATION SUMMARY:")
            logging.info(f"  Operation: Upload Google Drive folder to Cloudinary")
            logging.info(f"  Source folder ID: {folder_id}")
            logging.info(f"  Base folder name: {folder_name}")
            logging.info(f"  Recursive: {recursive}")
            logging.info(f"  Concurrent threads: {max_workers}")
            logging.info(f"  Total processing time: {elapsed_time:.2f} seconds")
            logging.info(f"  Success rate: {(successful/(successful+failed)*100):.1f}%" if (successful+failed) > 0 else "  Success rate: N/A")
            logging.info(f"{'='*60}")
            
            print(f"{'='*60}")
            print(f"📄 Detailed logs saved to: {log_file}")
            
        except Exception as e:
            print(f"Error writing CSV: {str(e)}")
            logging.error(f"Error writing CSV: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Check if command argument is provided
    if len(sys.argv) < 2:
        print("Usage: python googledrive_tocloudinary.py <command> [arguments]")
        print("\nCommands:")
        print("  list                                    : List all folders in your Google Drive")
        print("  shared                                  : List all files and folders shared with you + Shared Drives")
        print("  drives                                  : List only Shared Drives (Team Drives)")
        print("  upload <folder_id_or_url> [options]     : Upload images from a Google Drive folder")
        print("\nExamples:")
        print("  python googledrive_tocloudinary.py list")
        print("  python googledrive_tocloudinary.py shared")
        print("  python googledrive_tocloudinary.py drives")
        print("  # Using folder ID:")
        print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE")
        print("  # Using Google Drive URL:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2?usp=sharing'")
        print("  # With custom folder name:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2' my_custom_folder")
        print("  # With custom settings:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2' my_custom_folder 15 --no-recursive")
        print("\nUpload Arguments:")
        print("  folder_id_or_url : Google Drive folder ID OR full Google Drive URL")
        print("  destination_name : CUSTOM folder name for Cloudinary (optional, default: uses Drive folder name)")
        print("  threads          : Number of concurrent threads (optional, default: 10)")
        print("  --no-recursive   : Disable recursive scanning of subfolders (default: recursive enabled)")
        print("\nSupported URL Formats:")
        print("  ✓ https://drive.google.com/drive/folders/FOLDER_ID")
        print("  ✓ https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing")
        print("  ✓ https://drive.google.com/drive/u/0/folders/FOLDER_ID")
        print("  ✓ https://drive.google.com/folderview?id=FOLDER_ID")
        print("  ✓ FOLDER_ID (direct ID)")
        print("\nCloudinary Folder Organization:")
        print("  ✓ YES, folders are created automatically in Cloudinary")
        print("  ✓ You can specify custom destination folder name (2nd argument)")
        print("  ✓ Subfolders preserve structure: destination_name/subfolder1/subfolder2/image.jpg")
        print("  ✓ If no custom name provided, uses the original Google Drive folder name")
        print("\nReal-time Logging:")
        print("  ✓ Progress updates every 10 files in both console and log file")
        print("  ✓ Detailed per-folder statistics logged throughout operation")
        print("  ✓ Log files saved in data/log/ with timestamps")
        print("\nSetup:")
        print("  1. Download credentials.json from Google Cloud Console")
        print("  2. Enable Google Drive API in your Google Cloud project")
        print("  3. Place credentials.json in the root directory")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Handle "list" command to show all Google Drive folders
    if command == "list":
        print("Testing Google Drive connection...")
        is_connected, message = test_google_drive_connection()
        print(f"  {message}")
        
        if not is_connected:
            print("\n⚠️  Please check your Google Drive setup")
            sys.exit(1)
        
        print("  ✓ Google Drive verified\n")
        
        service = authenticate_google_drive()
        if not service:
            print("Failed to authenticate Google Drive")
            sys.exit(1)
        
        print("Scanning all folders in your Google Drive...\n")
        folders = list_all_google_drive_folders(service)
        
        print(f"\n{'='*60}")
        print(f"Total folders found: {len(folders)}")
        print(f"{'='*60}")
        
    # Handle "shared" command to show files shared with me
    elif command == "shared":
        print("Testing Google Drive connection...")
        is_connected, message = test_google_drive_connection()
        print(f"  {message}")
        
        if not is_connected:
            print("\n⚠️  Please check your Google Drive setup")
            sys.exit(1)
        
        print("  ✓ Google Drive verified\n")
        
        service = authenticate_google_drive()
        if not service:
            print("Failed to authenticate Google Drive")
            sys.exit(1)
        
        print("Scanning all shared content (shared files + Shared Drives)...\n")
        all_shared = list_all_shared_content(service)
        
        print(f"\n{'='*60}")
        print(f"📁 Total shared folders found: {len(all_shared['folders'])}")
        print(f"   - From files shared with you: {len([f for f in all_shared['folders'] if f.get('source') == 'shared_with_me'])}")
        print(f"   - From Shared Drives: {len([f for f in all_shared['folders'] if f.get('source') == 'shared_drive'])}")
        print(f"🖼️  Shared images found: {len(all_shared['files'])}")
        print(f"🚗 Shared Drives found: {len(all_shared['shared_drives'])}")
        print(f"{'='*60}")
        print("\n💡 Use any folder ID above with the 'upload' command")
        print("💡 Folders from Shared Drives should now be included!")
        
    # Handle "drives" command to show only Shared Drives
    elif command == "drives":
        print("Testing Google Drive connection...")
        is_connected, message = test_google_drive_connection()
        print(f"  {message}")
        
        if not is_connected:
            print("\n⚠️  Please check your Google Drive setup")
            sys.exit(1)
        
        print("  ✓ Google Drive verified\n")
        
        service = authenticate_google_drive()
        if not service:
            print("Failed to authenticate Google Drive")
            sys.exit(1)
        
        print("Scanning Shared Drives (Team Drives)...\n")
        shared_drives = list_shared_drives(service)
        total_folders = 0
        
        if shared_drives:
            print(f"\n📁 Scanning folders in each Shared Drive...\n")
            for drive in shared_drives:
                print(f"🚗 Shared Drive: {drive['name']} (ID: {drive['id']})")
                folders = list_folders_in_shared_drive(service, drive['id'], drive['name'])
                total_folders += len(folders)
                print()
        
        print(f"\n{'='*60}")
        print(f"🚗 Shared Drives found: {len(shared_drives)}")
        if shared_drives:
            print(f"📁 Total folders in Shared Drives: {total_folders}")
        print(f"{'='*60}")
        print("\n💡 Use any folder ID above with the 'upload' command")
        
    # Handle "upload" command
    elif command == "upload":
        if len(sys.argv) < 3:
            print("Error: Please provide a Google Drive folder ID or URL")
            print("\nUsage: python googledrive_tocloudinary.py upload <folder_id_or_url> [folder_name] [max_workers] [--no-recursive]")
            print("\nExamples:")
            print("  # Using folder ID:")
            print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE")
            print("  # Using Google Drive URL:")
            print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE?usp=sharing'")
            print("  # With custom folder name:")
            print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE' my_custom_folder 15")
            sys.exit(1)
        
        # Extract folder ID from URL or use as-is if it's already an ID
        folder_id_or_url = sys.argv[2]
        FOLDER_ID = extract_folder_id_from_url(folder_id_or_url)
        
        if not FOLDER_ID:
            print(f"❌ Error: Could not extract folder ID from: {folder_id_or_url}")
            print("\nSupported formats:")
            print("  - https://drive.google.com/drive/folders/FOLDER_ID")
            print("  - https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing")
            print("  - https://drive.google.com/drive/u/0/folders/FOLDER_ID")
            print("  - FOLDER_ID (direct ID)")
            sys.exit(1)
        
        print(f"🔍 Extracted folder ID: {FOLDER_ID}")
        
        # Validate folder access before proceeding
        print("🔐 Validating folder access...")
        service = authenticate_google_drive()
        if not service:
            print("❌ Failed to authenticate Google Drive")
            sys.exit(1)
        
        is_valid, folder_name_from_drive, error_msg = validate_folder_id(service, FOLDER_ID)
        if not is_valid:
            print(f"❌ {error_msg}")
            sys.exit(1)
        
        print(f"✅ Folder access confirmed: '{folder_name_from_drive}'")
        
        # Parse arguments
        args = sys.argv[3:]  # Get all arguments after folder_id
        FOLDER_NAME = None
        MAX_WORKERS = 10
        RECURSIVE = True  # Default to recursive
        
        # Process arguments
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--no-recursive":
                RECURSIVE = False
            elif arg == "--threads" and i + 1 < len(args):
                # Handle --threads argument
                try:
                    MAX_WORKERS = int(args[i + 1])
                    i += 1  # Skip the next argument as it's the thread count
                except ValueError:
                    print(f"Warning: Invalid thread count '{args[i + 1]}', using default: {MAX_WORKERS}")
                    i += 1
            elif arg.isdigit():
                MAX_WORKERS = int(arg)
            elif not arg.startswith('--'):
                # If it's not a flag and not a number, it's the folder name
                if FOLDER_NAME is None:
                    FOLDER_NAME = arg
            i += 1
        
        # Use the drive folder name if no custom name provided
        if not FOLDER_NAME:
            FOLDER_NAME = folder_name_from_drive
        
        print(f"📁 Source: {folder_id_or_url}")
        print(f"🆔 Folder ID: {FOLDER_ID}")
        print(f"📂 Destination folder name: {FOLDER_NAME}")
        print(f"🧵 Concurrent threads: {MAX_WORKERS}")
        print(f"🔄 Recursive scanning: {'Enabled' if RECURSIVE else 'Disabled'}")
        print()
        
        upload_gdrive_folder_to_cloudinary(FOLDER_ID, FOLDER_NAME, max_workers=MAX_WORKERS, recursive=RECURSIVE)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("\nAvailable commands: list, shared, drives, upload")
        print("Run 'python googledrive_tocloudinary.py' for usage information")
        sys.exit(1)