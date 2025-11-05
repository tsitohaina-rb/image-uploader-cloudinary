"""
Google Drive to Cloudinary Upload Script
========================================

Cross-platform compatible script for uploading images from Google Drive to Cloudinary.
Supports Windows, Linux, and macOS with:
- SSL-safe multiprocessing using ProcessPoolExecutor
- Cross-platform file paths with pathlib
- Platform-specific file locking (fcntl for Unix, msvcrt for Windows)
- Cloudinary folder management with user prompts
- Comprehensive caching and resume capabilities

Author: Tsitohaina
Date: November 2024
"""

import os
import csv
import sys
import json
import hashlib
import logging
import re
import ssl
import socket
import requests
import gc
import urllib.parse
import glob
import multiprocessing
import pickle
import platform
import unicodedata
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cloudinary
import cloudinary.uploader
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
import time

# Cross-platform file locking
try:
    import fcntl  # Unix/Linux/macOS
    HAS_FCNTL = True
except ImportError:
    try:
        import msvcrt  # Windows
        HAS_FCNTL = False
    except ImportError:
        HAS_FCNTL = None

def lock_file(file_handle):
    """Cross-platform file locking"""
    if HAS_FCNTL:
        # Unix/Linux/macOS
        import fcntl
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
    elif HAS_FCNTL is False:
        # Windows
        try:
            import msvcrt
            # Use blocking lock (1 = LK_LOCK equivalent)
            msvcrt.locking(file_handle.fileno(), 1, 1)
        except (ImportError, AttributeError, OSError):
            pass  # File locking not critical for this application
    # If neither works, continue without locking (not ideal but functional)

def unlock_file(file_handle):
    """Cross-platform file unlocking"""
    if HAS_FCNTL:
        # Unix/Linux/macOS
        import fcntl
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
    elif HAS_FCNTL is False:
        # Windows
        try:
            import msvcrt
            # Use unlock (0 = LK_UNLCK equivalent)
            msvcrt.locking(file_handle.fileno(), 0, 1)
        except (ImportError, AttributeError, OSError):
            pass  # Unlocking not critical
    # If neither works, no unlocking needed

def get_platform_info():
    """Get platform information for debugging"""
    return {
        'system': platform.system(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0],
        'file_locking': 'fcntl' if HAS_FCNTL else 'msvcrt' if HAS_FCNTL is False else 'none'
    }

# Setup directory structure - Use pathlib for cross-platform compatibility
DATA_DIR = Path('data')
CACHE_DIR = DATA_DIR / 'cache'
LOG_DIR = DATA_DIR / 'log'
OUTPUT_DIR = DATA_DIR / 'output'

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import USE_FILENAME, UNIQUE_FILENAME
except ImportError:
    # If config import fails, use default values
    USE_FILENAME = True
    UNIQUE_FILENAME = False

# Set socket timeout globally to 60 seconds (increased for better stability)
socket.setdefaulttimeout(60)

# SSL Context configuration for better multi-threading stability
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

# Threading-specific configuration
import threading
threading_local = threading.local()

# Retry decorator with exponential backoff for network issues
def retry_with_backoff(max_retries=5, backoff_factor=3, exceptions=(ssl.SSLError, TimeoutError, ConnectionError, OSError, Exception)):
    """
    Decorator to retry functions with exponential backoff on specified exceptions.
    Enhanced for SSL stability in multi-threading environments.
    
    Args:
        max_retries (int): Maximum number of retry attempts (increased to 5)
        backoff_factor (float): Multiplier for delay between retries (increased to 3)
        exceptions (tuple): Tuple of exception types to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # Last attempt, re-raise the exception
                        raise e
                    
                    # Progressive backoff with jitter to avoid thundering herd
                    base_wait = backoff_factor ** attempt
                    jitter = base_wait * 0.1 * (0.5 - hash(threading.current_thread().ident) % 1000 / 1000)
                    wait_time = base_wait + jitter
                    
                    error_type = type(e).__name__
                    thread_id = threading.current_thread().ident
                    logging.warning(f"  [Thread {thread_id}] Attempt {attempt + 1} failed with {error_type}: {str(e)}")
                    logging.info(f"  [Thread {thread_id}] Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# Configure HTTP session with proper connection pooling and timeouts
def create_robust_session():
    """
    Create a requests session with proper retry strategy and connection pooling.
    Optimized for multi-threading to avoid SSL errors.
    """
    session = requests.Session()
    
    # SSL-safe retry strategy with longer backoff
    retry_strategy = Retry(
        total=5,  # Increased retries
        backoff_factor=2,  # Longer backoff
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        raise_on_status=False,
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )
    
    # Conservative connection pooling to avoid SSL conflicts
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=1,  # One connection per thread
        pool_maxsize=1       # Single connection in pool
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def get_thread_session():
    """
    Get a thread-local session to avoid SSL conflicts between threads.
    """
    if not hasattr(threading_local, 'session'):
        threading_local.session = create_robust_session()
    return threading_local.session

# Google Drive API scopes - Updated to include full drive access for file permissions
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly'
]

# Configure logging
def setup_logging(folder_name):
    """Setup logging with timestamp and folder-specific log file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sanitize folder name for filename (replace problematic characters)
    safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('|', '_').replace('?', '_').replace('*', '_')
    
    log_filename = LOG_DIR / f'{safe_folder_name}_{timestamp}.log'
    
    # Configure logging with UTF-8 encoding for cross-platform compatibility
    file_handler = logging.FileHandler(str(log_filename), encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # On Windows, try to use UTF-8 encoding for console too
    if platform.system() == 'Windows':
        try:
            console_handler.stream = open(console_handler.stream.fileno(), mode='w', encoding='utf-8', buffering=1)
        except:
            pass  # Fall back to default if UTF-8 setup fails
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    return str(log_filename)

def is_cloudinary_url(url):
    """Check if a URL is a Cloudinary URL"""
    if not url or not isinstance(url, str):
        return False
    return 'cloudinary.com' in url.lower()

def get_current_format(url):
    """Extract the current format from a Cloudinary URL"""
    if not is_cloudinary_url(url):
        return None
    
    # Parse the URL path to find format
    parsed = urllib.parse.urlparse(url)
    path_parts = parsed.path.split('/')
    
    # Look for format parameter (f_xxx) or file extension
    for part in path_parts:
        if part.startswith('f_'):
            return part[2:]  # Remove 'f_' prefix
    
    # Check file extension in the last part
    if path_parts:
        last_part = path_parts[-1]
        if '.' in last_part:
            return last_part.split('.')[-1].lower()
    
    return None

def convert_cloudinary_url_to_jpg(url):
    """Convert a Cloudinary URL to JPG format"""
    if not is_cloudinary_url(url):
        return url
    
    # Parse the URL
    parsed = urllib.parse.urlparse(url)
    path_parts = parsed.path.split('/')
    
    # Find the upload part and insert format transformation
    if '/image/upload/' in parsed.path:
        upload_index = None
        for i, part in enumerate(path_parts):
            if part == 'upload':
                upload_index = i
                break
        
        if upload_index is not None:
            # Insert JPG format transformation after 'upload'
            path_parts.insert(upload_index + 1, 'f_jpg')
            
            # Rebuild the path
            new_path = '/'.join(path_parts)
            
            # Rebuild the URL
            new_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                new_path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            return new_url
    
    return url  # Return original if transformation failed

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
    
    def __init__(self, folder_path: str, folder_name: str | None = None):
        """Initialize cache for a specific Google Drive folder"""
        self.folder_path = folder_path
        
        # Create cache file name consistent with log and output files
        if folder_name:
            # Sanitize folder name for filename (same as done for log filename)
            safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('|', '_').replace('?', '_').replace('*', '_').strip()
            
            # Look for existing cache files first
            cache_pattern = str(CACHE_DIR / f'gdrive_upload_cache_{safe_folder_name}_*.json')
            existing_caches = glob.glob(cache_pattern)
            
            if existing_caches:
                # Use the most recent existing cache file
                self.cache_file = Path(max(existing_caches, key=os.path.getctime))
                print(f"🔄 RESUMING: Found existing cache file: {self.cache_file.name}")
            else:
                # Create new cache file with current timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.cache_file = CACHE_DIR / f'gdrive_upload_cache_{safe_folder_name}_{timestamp}.json'
                print(f"🆕 STARTING: Creating new cache file: {self.cache_file.name}")
        else:
            # Fallback to hash-based naming if no folder name provided
            folder_hash = hashlib.md5(folder_path.encode()).hexdigest()
            self.cache_file = CACHE_DIR / f'gdrive_upload_cache_{folder_hash}.json'
        
        self.lock = Lock()
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load the cache from file"""
        try:
            if self.cache_file.exists():
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
    token_file = CACHE_DIR / 'token.pickle'
    
    # The file token.pickle stores the user's access and refresh tokens.
    if token_file.exists():
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            credentials_file = 'credentials-regardbeauty.json'  # Use the new credentials file
            if not os.path.exists(credentials_file):
                print("Error: credentials-regardbeauty.json not found!")
                print("Please download credentials-regardbeauty.json from Google Cloud Console:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a project or select existing one")
                print("3. Enable Google Drive API")
                print("4. Create credentials (OAuth 2.0 Client ID)")
                print("5. Download credentials-regardbeauty.json and place it in the root directory")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

@retry_with_backoff(max_retries=3, backoff_factor=2, exceptions=(ssl.SSLError, TimeoutError, ConnectionError, Exception))
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

@retry_with_backoff(max_retries=3, backoff_factor=2, exceptions=(ssl.SSLError, TimeoutError, ConnectionError, Exception))
def check_file_permissions(service, file_id, filename):
    """
    Check and manage file permissions with retry logic.
    Returns public_permission_id if permission was granted, None otherwise.
    """
    permissions = service.permissions().list(
        fileId=file_id,
        supportsAllDrives=True
    ).execute()
    
    # Check if file already has public read access
    has_public_access = any(
        perm.get('type') == 'anyone' and 'read' in perm.get('role', '')
        for perm in permissions.get('permissions', [])
    )
    
    if not has_public_access:
        # Temporarily grant public read access
        logging.info(f"  Granting temporary public access: {filename}")
        public_permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        permission_result = service.permissions().create(
            fileId=file_id,
            body=public_permission,
            supportsAllDrives=True
        ).execute()
        public_permission_id = permission_result.get('id')
        logging.info(f"  Temporary public access granted: {filename}")
        return public_permission_id
    else:
        logging.info(f"  File already has public access: {filename}")
        return None

@retry_with_backoff(max_retries=3, backoff_factor=2, exceptions=(ssl.SSLError, TimeoutError, ConnectionError, Exception))
def get_file_download_url(service, file_id, filename):
    """
    Get file download URL with retry logic.
    """
    # Try to get webContentLink first
    file_metadata = service.files().get(
        fileId=file_id, 
        fields='webContentLink,webViewLink',
        supportsAllDrives=True
    ).execute()
    
    if 'webContentLink' in file_metadata:
        download_url = file_metadata['webContentLink']
        logging.info(f"  Using webContentLink for: {filename}")
        return download_url
    else:
        # Fallback to direct download URL
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        logging.info(f"  Using fallback URL for: {filename}")
        return download_url

def save_progress(progress_counter, folder_id, folder_name, total_files, timestamp):
    """
    Save current progress to a JSON file for resuming interrupted uploads.
    """
    progress_file = CACHE_DIR / f'progress_{folder_id}_{timestamp}.json'
    
    try:
        progress_data = {
            'folder_id': folder_id,
            'folder_name': folder_name,
            'timestamp': timestamp,
            'total_files': total_files,
            'uploaded': progress_counter['uploaded'],
            'failed': progress_counter['failed'],
            'skipped': progress_counter['skipped'],
            'last_update': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        logging.info(f"Progress saved to: {progress_file}")
        
    except Exception as e:
        logging.warning(f"Could not save progress: {e}")

def load_previous_progress(folder_id):
    """
    Load previous progress if available.
    """
    progress_pattern = str(CACHE_DIR / f'progress_{folder_id}_*.json')
    import glob
    
    progress_files = glob.glob(progress_pattern)
    if not progress_files:
        return None
        
    # Get the most recent progress file
    latest_file = max(progress_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            progress_data = json.load(f)
            
        logging.info(f"Found previous progress: {progress_data['uploaded']} uploaded, {progress_data['failed']} failed, {progress_data['skipped']} skipped")
        return progress_data
        
    except Exception as e:
        logging.warning(f"Could not load progress from {latest_file}: {e}")
        return None

def check_cloudinary_folder_exists(folder_name):
    """
    Check if a folder exists in Cloudinary using the fast folders API.
    
    Args:
        folder_name (str): The folder name to check
        
    Returns:
        tuple: (exists, resource_count, sample_urls)
    """
    try:
        import cloudinary.api
        
        # Use the fast folders API to check if folder exists
        try:
            # Get root level folders first
            result = cloudinary.api.root_folders()
            root_folders = result.get('folders', [])
            
            # Check if the folder name exists in root folders (case-insensitive exact match)
            folder_name_lower = folder_name.lower()
            for folder in root_folders:
                if folder['name'].lower() == folder_name_lower:
                    # Folder exists! Now get some sample resources for display
                    try:
                        resources_result = cloudinary.api.resources(
                            type="upload",
                            prefix=f"{folder['name']}/",
                            max_results=10  # Just need to know how many exist
                        )
                        resources = resources_result.get('resources', [])
                        sample_urls = [res['secure_url'] for res in resources[:3]]  # Show up to 3 samples
                        return True, len(resources), sample_urls
                    except:
                        # Folder exists but maybe has no files or access issue
                        return True, 0, []
            
            # Folder not found in root folders
            return False, 0, []
            
        except Exception as e:
            print(f"Warning: Could not use folders API, falling back to resource search: {e}")
            # Fallback to old method if folders API fails
            result = cloudinary.api.resources(
                type="upload",
                prefix=f"{folder_name}/",
                max_results=10  # Just need to know if any exist
            )
            
            resources = result.get('resources', [])
            exists = len(resources) > 0
            sample_urls = [res['secure_url'] for res in resources[:3]]  # Show up to 3 samples
            
            return exists, len(resources), sample_urls
        
    except Exception as e:
        print(f"Warning: Could not check Cloudinary folder existence: {e}")
        return False, 0, []

def list_cloudinary_folders():
    """
    List all folders in Cloudinary using the efficient folders API.
    
    Returns:
        dict: Dictionary with folder information
    """
    try:
        import cloudinary.api
        
        print("📊 Scanning Cloudinary for existing folders...")
        
        # Use the folders API for much faster folder listing
        all_folders = {}
        
        try:
            # Get root level folders first
            result = cloudinary.api.root_folders()
            root_folders = result.get('folders', [])
            
            print(f"✅ Found {len(root_folders)} root-level folders")
            
            # Just store root folder info without any recursive processing
            for folder in root_folders:
                folder_name = folder['name']
                all_folders[folder_name] = {
                    'file_count': 0,  # Don't count files for speed
                    'created_at': folder.get('created_at', ''),
                    'last_updated': folder.get('created_at', '')
                }
            
            # Also check for resources in the root (no folder) - skip for speed
            # We'll just show folders, not count root files
            
        except Exception as e:
            print(f"Warning: Could not use folders API, falling back to resource scanning: {e}")
            # Fallback to the old method if folders API fails
            return list_cloudinary_folders_legacy()
        
        print(f"📁 Found {len(all_folders)} folders in Cloudinary")
        print()
        
        # Display results
        if all_folders:
            print("📋 Cloudinary Folder Structure:")
            print("=" * 80)
            
            # Sort folders for better display (root first, then alphabetically)
            # Sort folders alphabetically
            sorted_folders = sorted(all_folders.items())
            
            for folder_path, info in sorted_folders:
                created_date = info["created_at"][:10] if info["created_at"] else "Unknown"
                print(f"📂 {folder_path:60} │ Created: {created_date}")
            
            print("=" * 80)
            print(f"💡 Total: {len(all_folders)} folders")
            print("💡 Use 'upload' command with any of these folder names to add more images")            
        else:
            print("📭 No folders found in Cloudinary (account appears to be empty)")
            print("💡 Upload some images first using the 'upload' command")
        
        return all_folders
        
    except Exception as e:
        print(f"❌ Error accessing Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def list_cloudinary_folders_legacy():
    """
    Legacy method: List all folders by analyzing all resources (slower but comprehensive).
    Used as fallback when folders API is not available.
    
    Returns:
        dict: Dictionary with folder information
    """
    try:
        import cloudinary.api
        
        print("📊 Using legacy method: Scanning all Cloudinary resources...")
        
        # Get all resources to analyze folder structure
        all_folders = {}
        next_cursor = None
        total_resources = 0
        
        while True:
            try:
                # Get resources in batches
                if next_cursor:
                    result = cloudinary.api.resources(
                        type="upload",
                        resource_type="image",
                        max_results=500,  # Maximum allowed per request
                        next_cursor=next_cursor
                    )
                else:
                    result = cloudinary.api.resources(
                        type="upload", 
                        resource_type="image",
                        max_results=500
                    )
                
                resources = result.get('resources', [])
                total_resources += len(resources)
                
                # Analyze each resource for folder structure
                for resource in resources:
                    public_id = resource.get('public_id', '')
                    
                    # Extract folder path from public_id
                    if '/' in public_id:
                        # Split the public_id to get folder parts
                        parts = public_id.split('/')
                        filename = parts[-1]  # Last part is the filename
                        folder_parts = parts[:-1]  # All parts except the last
                        
                        # Build folder hierarchy
                        current_path = ""
                        for i, part in enumerate(folder_parts):
                            if i == 0:
                                current_path = part
                            else:
                                current_path = f"{current_path}/{part}"
                            
                            if current_path not in all_folders:
                                all_folders[current_path] = {
                                    'file_count': 0,
                                    'created_at': resource.get('created_at', ''),
                                    'last_updated': resource.get('created_at', '')
                                }
                            
                            # Update folder info
                            folder_info = all_folders[current_path]
                            folder_info['file_count'] += 1
                            
                            # Update timestamps
                            resource_date = resource.get('created_at', '')
                            if resource_date:
                                if not folder_info['created_at'] or resource_date < folder_info['created_at']:
                                    folder_info['created_at'] = resource_date
                                if not folder_info['last_updated'] or resource_date > folder_info['last_updated']:
                                    folder_info['last_updated'] = resource_date
                    else:
                        # File in root directory
                        root_key = "(root)"
                        if root_key not in all_folders:
                            all_folders[root_key] = {
                                'file_count': 0,
                                'created_at': resource.get('created_at', ''),
                                'last_updated': resource.get('created_at', '')
                            }
                        
                        folder_info = all_folders[root_key]
                        folder_info['file_count'] += 1
                
                # Check if there are more resources
                next_cursor = result.get('next_cursor')
                if not next_cursor:
                    break
                    
                # Progress update for large accounts
                if total_resources % 1000 == 0:
                    print(f"  Processed {total_resources} resources, found {len(all_folders)} folders...")
                    
            except Exception as e:
                print(f"Error fetching resources: {e}")
                break
        
        print(f"✅ Legacy scan complete: {total_resources} resources analyzed")
        print(f"📁 Found {len(all_folders)} folders in Cloudinary")
        print()
        
        # Display results
        if all_folders:
            print("📋 Cloudinary Folder Structure:")
            print("=" * 80)
            
            # Sort folders for better display (root first, then alphabetically)
            sorted_folders = sorted(all_folders.items(), key=lambda x: (x[0] != "(root)", x[0]))
            
            for folder_path, info in sorted_folders:
                file_count = info['file_count']
                created_at = info.get('created_at', '')
                
                # Format dates
                try:
                    from datetime import datetime
                    if created_at:
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        created_str = created_date.strftime('%Y-%m-%d %H:%M')
                    else:
                        created_str = "Unknown"
                except:
                    created_str = "Unknown"
                
                # Display folder info
                if folder_path == "(root)":
                    print(f"📁 {folder_path:<50} │ {file_count:>6} files │ Created: {created_str}")
                else:
                    # Calculate indentation based on folder depth
                    depth = folder_path.count('/')
                    indent = "  " * depth
                    folder_name = folder_path.split('/')[-1]
                    print(f"📁 {indent}{folder_name:<50} │ {file_count:>6} files │ Created: {created_str}")
            
            print("=" * 80)
            print(f"💡 Total: {len(all_folders)} folders with {total_resources} images")
            print("💡 Use 'upload' command with any of these folder names to add more images")
            
        else:
            print("📭 No folders found in Cloudinary (account appears to be empty)")
            print("💡 Upload some images first using the 'upload' command")
        
        return all_folders
        
    except Exception as e:
        print(f"❌ Error accessing Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def search_cloudinary_folder(search_term):
    """
    Search for specific folders in Cloudinary by name (case-insensitive, partial match).
    Uses the fast folders API instead of scanning all resources.
    
    Args:
        search_term (str): The folder name or part of folder name to search for
        
    Returns:
        dict: Dictionary with matching folder information
    """
    try:
        import cloudinary.api
        
        print(f"🔍 Searching Cloudinary for folders matching: '{search_term}'")
        
        # Use the fast folders API to get all root folders
        all_folders = {}
        
        try:
            # Get root level folders first
            result = cloudinary.api.root_folders()
            root_folders = result.get('folders', [])
            
            print(f"✅ Found {len(root_folders)} root-level folders to search")
            
            # Just store root folder info without any recursive processing
            for folder in root_folders:
                folder_name = folder['name']
                all_folders[folder_name] = {
                    'file_count': 0,  # Don't count files for speed
                    'created_at': folder.get('created_at', ''),
                    'last_updated': folder.get('created_at', '')
                }
            
        except Exception as e:
            print(f"Warning: Could not use folders API: {e}")
            return {}
        
        # Filter folders that match the search term (case-insensitive, partial match)
        search_term_lower = search_term.lower()
        matching_folders = {}
        
        for folder_path, info in all_folders.items():
            if search_term_lower in folder_path.lower():
                matching_folders[folder_path] = info
        
        print(f"✅ Search complete: {len(all_folders)} folders scanned")
        print()
        
        # Display results
        if matching_folders:
            print(f"🎯 Found {len(matching_folders)} folder(s) matching '{search_term}':")
            print("=" * 80)
            
            # Sort folders alphabetically
            sorted_folders = sorted(matching_folders.items())
            
            for folder_path, info in sorted_folders:
                created_date = info['created_at'][:10] if info['created_at'] else 'Unknown'
                print(f"📂 {folder_path:<50} │ Created: {created_date}")
            
            print("=" * 80)
            print(f"💡 Found {len(matching_folders)} matching folders")
            print("💡 Use any of these folder names with the 'upload' command to add more images")
            
        else:
            print(f"📭 No folders found matching '{search_term}'")
            print("💡 Try a different search term or use 'cloudinary' command to see all folders")
        
        return matching_folders
        
    except Exception as e:
        print(f"❌ Error searching Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def prompt_folder_action(folder_name, resource_count, sample_urls):
    """
    Prompt user for action when folder already exists in Cloudinary.
    
    Args:
        folder_name (str): The existing folder name
        resource_count (int): Number of existing resources
        sample_urls (list): Sample URLs from existing folder
        
    Returns:
        tuple: (action, new_folder_name) where action is 'merge', 'rename', or 'cancel'
    """
    print(f"\n⚠️  Folder '{folder_name}' already exists in Cloudinary!")
    print(f"   📊 Contains {resource_count} existing images")
    
    if sample_urls:
        print(f"   🖼️  Sample images:")
        for i, url in enumerate(sample_urls[:3], 1):
            filename = url.split('/')[-1]
            print(f"      {i}. {filename}")
        if resource_count > 3:
            print(f"      ... and {resource_count - 3} more")
    
    print(f"\n🤔 What would you like to do?")
    print(f"   1. MERGE - Add new images to existing folder (skip duplicates)")
    print(f"   2. RENAME - Use a different folder name")
    print(f"   3. CANCEL - Stop the upload")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print(f"✅ Will merge with existing folder '{folder_name}'")
            return 'merge', folder_name
        elif choice == '2':
            while True:
                new_name = input(f"\n📝 Enter new folder name: ").strip()
                if new_name:
                    # Check if the new name also exists
                    exists, count, urls = check_cloudinary_folder_exists(new_name)
                    if exists:
                        print(f"⚠️  Folder '{new_name}' also exists ({count} images)")
                        retry = input("Try a different name? (y/n): ").strip().lower()
                        if retry == 'y':
                            continue
                        else:
                            return 'merge', new_name
                    else:
                        print(f"✅ Will use new folder name '{new_name}'")
                        return 'rename', new_name
                else:
                    print("❌ Folder name cannot be empty")
        elif choice == '3':
            print("❌ Upload cancelled")
            return 'cancel', None
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3")

def sanitize_cloudinary_public_id(text):
    """
    Sanitize text for use in Cloudinary public_id.
    Cloudinary public_ids can only contain: a-z, A-Z, 0-9, -, _, /
    """
    if not text:
        return text
    
    # First, handle accents by converting to ASCII equivalents
    import unicodedata
    # Normalize and convert accented characters to ASCII
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Replace problematic characters
    # & becomes 'and'
    text = text.replace('&', 'and')
    # Spaces become underscores
    text = re.sub(r'\s+', '_', text)
    # Remove any remaining invalid characters, keep only allowed ones
    text = re.sub(r'[^a-zA-Z0-9\-_/]', '', text)
    # Clean up multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores and slashes, but preserve the content
    text = text.strip('_/')
    
    # Ensure we don't return empty string
    if not text:
        text = "uploaded_files"
    
    return text

def process_worker_upload(args):
    """
    Worker function for multiprocessing upload.
    This runs in a separate process, avoiding SSL conflicts.
    """
    file_info, base_folder_name, credentials_file, cache_file = args
    
    # Each process needs its own service and cache instances
    try:
        # Initialize Cloudinary in the worker process
        import cloudinary
        import cloudinary.uploader
        from dotenv import load_dotenv
        from googleapiclient.http import MediaIoBaseDownload
        import io
        load_dotenv()
        
        # Configure Cloudinary in the worker process
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )
        
        # Re-authenticate Google Drive in worker process
        creds = None
        token_file = Path('data/cache') / 'token.pickle'
        
        if token_file.exists():
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
        
        service = build('drive', 'v3', credentials=creds)
        
        # Load cache in worker process
        cache_data = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        
        file_id = file_info['id']
        filename = file_info['name']
        
        # Check if already uploaded
        if file_id in cache_data.get('successful_uploads', {}):
            return {
                'local_filename': os.path.splitext(filename)[0],
                'cloudinary_url': cache_data['successful_uploads'][file_id]['cloudinary_url'],
                'status': 'skipped',
                'public_id': cache_data['successful_uploads'][file_id].get('public_id', ''),
                'filename': filename
            }
        
        # Sanitize folder name for Cloudinary
        cloudinary_folder = sanitize_cloudinary_public_id(base_folder_name.strip())
        
        # Ensure we have a valid folder name
        if not cloudinary_folder:
            cloudinary_folder = "uploaded_files"  # Fallback folder name
        
        # Get file permissions and download URL
        try:
            # For multiprocessing, use authenticated download directly for shared files
            # This is more reliable than trying URL methods first
            
            # Prepare upload parameters first
            file_stem, ext = os.path.splitext(filename)
            file_stem = file_stem.strip()
            original_extension = ext.lower().replace('.', '') if ext else 'jpg'
            
            # Create complete public_id with sanitized folder name
            complete_public_id = f"{cloudinary_folder}/{file_stem}"
            
            # Download file content using Google Drive API (most reliable for shared files)
            import io
            from googleapiclient.http import MediaIoBaseDownload
            
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            
            # Upload the downloaded content to Cloudinary
            response = cloudinary.uploader.upload(
                file_content,
                public_id=complete_public_id,
                use_filename=False,
                unique_filename=False,
                overwrite=True,
                format=original_extension,
                resource_type="image",
                timeout=120
            )
            
            # Update cache with proper multiprocessing safety
            success_data = {
                'timestamp': datetime.now().isoformat(),
                'cloudinary_url': response['secure_url'],
                'public_id': response.get('public_id', ''),
                'filename': filename
            }
            
            # Save to cache with proper file locking and race condition handling
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Try to open existing file for update
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r+') as f:
                            # Lock the entire file for exclusive access
                            lock_file(f)
                            
                            try:
                                # Read current data
                                f.seek(0)
                                try:
                                    cache_data = json.load(f)
                                except json.JSONDecodeError:
                                    # File is corrupted, start fresh
                                    cache_data = {
                                        'successful_uploads': {},
                                        'failed_uploads': {},
                                        'last_run': datetime.now().isoformat()
                                    }
                                
                                # Ensure structure exists
                                if 'successful_uploads' not in cache_data:
                                    cache_data['successful_uploads'] = {}
                                
                                # Update with new data
                                cache_data['successful_uploads'][file_id] = success_data
                                cache_data['last_run'] = datetime.now().isoformat()
                                
                                # Write back atomically
                                f.seek(0)
                                json.dump(cache_data, f, indent=2)
                                f.truncate()
                                break  # Success, exit retry loop
                            finally:
                                unlock_file(f)
                    else:
                        # Create new file with proper locking
                        # Use a temporary file to ensure atomic creation
                        temp_cache_file = str(cache_file) + '.tmp'
                        cache_data = {
                            'successful_uploads': {file_id: success_data},
                            'failed_uploads': {},
                            'last_run': datetime.now().isoformat()
                        }
                        
                        with open(temp_cache_file, 'w') as f:
                            lock_file(f)
                            try:
                                json.dump(cache_data, f, indent=2)
                            finally:
                                unlock_file(f)
                        
                        # Atomic rename
                        os.rename(temp_cache_file, cache_file)
                        break  # Success, exit retry loop
                        
                except (OSError, IOError, json.JSONDecodeError) as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed, log but don't crash
                        print(f"WARNING: Failed to update cache for {filename} after {max_retries} attempts: {e}")
                        break
                    else:
                        # Wait a bit before retrying
                        import time
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
            return {
                'local_filename': file_stem,
                'cloudinary_url': response['secure_url'],
                'status': 'success',
                'public_id': response.get('public_id', ''),
                'filename': filename
            }
            
        except Exception as e:
            return {
                'local_filename': os.path.splitext(filename)[0],
                'cloudinary_url': 'UPLOAD_FAILED',
                'status': 'failed',
                'error': str(e),
                'filename': filename
            }
    
    except Exception as e:
        return {
            'local_filename': os.path.splitext(file_info.get('name', 'unknown'))[0],
            'cloudinary_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': str(e),
            'filename': file_info.get('name', 'unknown')
        }

def upload_single_image_from_gdrive(service, file_info, base_folder_name, cache):
    """
    Cloud-to-cloud: Create a temporary public link and let Cloudinary fetch it directly.
    Enhanced with thread-local sessions to avoid SSL conflicts.
    """
    file_id = file_info['id']
    filename = file_info['name']
    folder_path = file_info.get('folder_path', '')
    
    # Get thread-local session for this upload
    session = get_thread_session()
    
    # Create the full Cloudinary folder path with comprehensive whitespace cleaning
    # Use only the base folder name for a flatter structure
    cloudinary_folder = base_folder_name.strip()
    
    # Comprehensive whitespace cleaning: 
    # 1. Strip leading/trailing spaces
    # 2. Remove trailing forward slashes
    # 3. Clean up spaces around forward slashes
    # 4. Remove multiple consecutive spaces
    cloudinary_folder = cloudinary_folder.strip().rstrip('/')
    
    # Clean spaces around forward slashes: " / " becomes "/"
    cloudinary_folder = re.sub(r'\s*/\s*', '/', cloudinary_folder)
    
    # Remove multiple consecutive spaces
    cloudinary_folder = re.sub(r'\s+', ' ', cloudinary_folder)
    
    # Final strip to ensure no trailing whitespace
    cloudinary_folder = cloudinary_folder.strip()
    
    # Debug logging to see exact folder path
    logging.info(f"  Final cloudinary folder path: '{cloudinary_folder}' (length: {len(cloudinary_folder)})")
    if cloudinary_folder.endswith(' '):
        logging.warning(f"  WARNING: Folder path still ends with whitespace!")
        cloudinary_folder = cloudinary_folder.rstrip()
        logging.info(f"  Cleaned folder path: '{cloudinary_folder}'")
    
    thread_id = threading.current_thread().ident
    logging.info(f"[Thread {thread_id}] Processing: {filename} (ID: {file_id}) -> {cloudinary_folder}")
    
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
        original_url = cached_data['cloudinary_url']
        
        # Generate JPG URL for cached result
        jpg_url = original_url
        if is_cloudinary_url(original_url):
            current_format = get_current_format(original_url)
            if current_format == 'png':
                jpg_url = convert_cloudinary_url_to_jpg(original_url)
        
        return {
            'local_filename': os.path.splitext(filename)[0],
            'cloudinary_url': original_url,
            'jpg_url': jpg_url,
            'status': 'skipped',
            'public_id': cached_data.get('public_id', ''),
            'folder_path': folder_path
        }

    # Initialize permission tracking
    public_permission_id = None
    
    try:
        logging.info(f"START TRANSFER: {filename} -> {cloudinary_folder}")
        
        # 1) Check if file already has public access
        logging.info(f"  Checking file permissions: {filename}")
        
        try:
            public_permission_id = check_file_permissions(service, file_id, filename)
        except Exception as perm_error:
            logging.warning(f"  Could not manage permissions for {filename}: {perm_error}")
            public_permission_id = None
            
        # 2) Get the download URL
        logging.info(f"  Getting download URL: {filename}")
        
        try:
            download_url = get_file_download_url(service, file_id, filename)
        except Exception as url_error:
            logging.warning(f"  Error getting URL for {filename}: {url_error}")
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            logging.info(f"  Using fallback URL for: {filename}")

        # 3) Derive filename / extension for Cloudinary options
        file_stem, ext = os.path.splitext(filename)
        # Clean the file stem to avoid whitespace issues
        file_stem = file_stem.strip()
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        logging.info(f"  Uploading to Cloudinary as: {file_stem}.{original_extension}")

        # 4) Cloudinary pulls the file directly from Google Drive with enhanced error handling
        try:
            # Combine folder path and filename for the complete public_id
            complete_public_id = f"{cloudinary_folder}/{file_stem}"
            
            response = cloudinary.uploader.upload(
                download_url,
                public_id=complete_public_id,   # use complete path as public_id
                use_filename=False,             # don't use remote URL name
                unique_filename=False,          # keep stable public_id
                overwrite=True,                 # allow re-runs to overwrite
                format=original_extension,      # keep original extension
                resource_type="image",          # or "auto" if you might have videos/svg/others
                timeout=120                     # Extended timeout for Cloudinary
            )
        except Exception as cloudinary_error:
            # If Cloudinary upload fails, add extra context
            error_msg = f"Cloudinary upload failed: {str(cloudinary_error)}"
            logging.error(f"[Thread {thread_id}] {error_msg}")
            raise Exception(error_msg)

        result = {
            'local_filename': file_stem,
            'cloudinary_url': response['secure_url'],
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'filename': filename,
            'folder_path': folder_path
        }
        
        # Add JPG URL for successful upload
        original_url = response['secure_url']
        if is_cloudinary_url(original_url):
            current_format = get_current_format(original_url)
            if current_format == 'png':
                result['jpg_url'] = convert_cloudinary_url_to_jpg(original_url)
                logging.info(f"  [JPG] Generated JPG URL: {filename}")
            else:
                result['jpg_url'] = original_url  # Already JPG or other format
        else:
            result['jpg_url'] = original_url  # Not a Cloudinary URL

        # 5) Cache + progress + real-time logging
        cache.mark_uploaded(file_id, result)
        with progress_lock:
            progress_counter['uploaded'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SUCCESS: {filename} -> {result['cloudinary_url']}")
            
            # Real-time progress logging every 10 files or at completion
            if current % 10 == 0 or current == progress_counter['total']:
                progress_msg = (f"PROGRESS UPDATE: {current}/{progress_counter['total']} "
                               f"(✓Success: {progress_counter['uploaded']}, ❌Failed: {progress_counter['failed']}, "
                               f"⏭️Skipped: {progress_counter['skipped']})")
                logging.info(progress_msg)
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")
                
                # Force garbage collection every 10 files to prevent memory issues
                gc.collect()

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
                
                # Force garbage collection every 10 files to prevent memory issues
                gc.collect()

        file_stem = os.path.splitext(filename)[0]
        cache.mark_failed(file_id, error_message)
        
        return {
            'local_filename': file_stem,
            'cloudinary_url': 'UPLOAD_FAILED',
            'jpg_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message,
            'folder_path': folder_path
        }
    
    finally:
        # 6) Cleanup: Remove temporary public permission if we added one
        if public_permission_id:
            try:
                service.permissions().delete(
                    fileId=file_id,
                    permissionId=public_permission_id,
                    supportsAllDrives=True
                ).execute()
                logging.info(f"  Removed temporary public access from: {filename}")
            except Exception as cleanup_error:
                logging.warning(f"  Could not remove temporary permission from {filename}: {cleanup_error}")

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
                    'folder_path': parent_path.strip() if parent_path else '',  # Clean the folder path
                    'folder_name': folder_name.strip() if folder_name else ''
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
                    # Clean subfolder name to remove any whitespace
                    clean_subfolder_name = subfolder['name'].strip()
                    subfolder_path = f"{parent_path.strip()}/{clean_subfolder_name}" if parent_path else clean_subfolder_name
                    print(f"  📁 Scanning subfolder: {subfolder_path}")
                    
                    # Recursively get images from subfolder
                    subfolder_images = get_images_from_gdrive_folder(
                        service, 
                        subfolder['id'], 
                        recursive=True, 
                        parent_path=subfolder_path,
                        folder_name=clean_subfolder_name
                    )
                    image_files.extend(subfolder_images)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{folder_id}': {str(e)}")
    
    return image_files

def upload_gdrive_folder_to_cloudinary(folder_id, folder_name=None, max_workers=5, recursive=True):
    """
    Upload images from a Google Drive folder to Cloudinary using multi-threading.
    Supports resuming interrupted uploads through caching and recursive subfolder scanning.
    Enhanced with SSL-safe threading for better stability.
    
    Args:
        folder_id (str): Google Drive folder ID
        folder_name (str): Optional custom folder name for Cloudinary (default: uses Drive folder name)
        max_workers (int): Number of concurrent upload threads (default: 5, optimized for SSL stability)
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
            folder_name = folder_info['name'].strip()  # Remove leading/trailing whitespace
        except Exception as e:
            print(f"Error getting folder name: {e}")
            folder_name = f"gdrive_folder_{folder_id}"
    else:
        folder_name = folder_name.strip()  # Also trim user-provided folder names
    
    # Ensure folder_name is not None
    if not folder_name:
        folder_name = f"gdrive_folder_{folder_id}"
    
    # Check if folder already exists in Cloudinary and get user preference
    print(f"🔍 Checking if folder '{folder_name}' exists in Cloudinary...")
    exists, resource_count, sample_urls = check_cloudinary_folder_exists(folder_name)
    
    if exists:
        action, final_folder_name = prompt_folder_action(folder_name, resource_count, sample_urls)
        
        if action == 'cancel':
            print("Upload cancelled by user")
            return
        elif action == 'rename':
            folder_name = final_folder_name or f"gdrive_folder_{folder_id}"
            print(f"📂 Using folder name: '{folder_name}'")
        elif action == 'merge':
            print(f"📂 Merging with existing folder: '{folder_name}'")
    else:
        print(f"✅ Folder '{folder_name}' does not exist - will be created")
        print(f"📁 New folder will be created at: /cloudinary/{folder_name}/")
    
    print()  # Add spacing before next section
    
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
        logging.info(f"  [FOLDER] {folder_path}: {count} images -> will be uploaded to: {cloudinary_path}")
    logging.info("")
    
    # Generate CSV filename based on folder name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sanitize folder name for CSV filename (same as done for log filename)
    safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('|', '_').replace('?', '_').replace('*', '_').strip()
    
    output_csv = OUTPUT_DIR / f"{safe_folder_name}_{timestamp}.csv"
    
    # Initialize upload cache
    cache = UploadCache(folder_id, folder_name)
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
    logging.info(f"Using {max_workers} concurrent processes for faster upload")
    logging.info(f"Output will be saved to: {output_csv}")
    logging.info(f"Log file: {log_file}\n")
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for concurrent uploads with SSL isolation
    # Each process gets its own SSL context, avoiding threading conflicts
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for worker processes
        credentials_file = 'credentials-regardbeauty.json'  # Or appropriate credentials file
        cache_file = cache.cache_file
        
        # Create list of arguments for each upload task
        upload_tasks = []
        for img in image_files:
            upload_tasks.append((img, folder_name, credentials_file, cache_file))
        
        # Submit all upload tasks
        future_to_image = {}
        for i, task_args in enumerate(upload_tasks):
            future = executor.submit(process_worker_upload, task_args)
            future_to_image[future] = task_args[0]  # Store the image info
        
        # Collect results as they complete
        completed_files = set()  # Track completed files to avoid missing any
        expected_total = len(upload_tasks)
        
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                img_info = future_to_image[future]
                completed_files.add(img_info['id'])  # Track completion
                results.append(result)
                
                # Update progress counters
                if result['status'] == 'success':
                    progress_counter['uploaded'] += 1
                    print(f"✅ [{len(results)}/{progress_counter['total']}] {result['filename']} → {result['cloudinary_url']}")
                elif result['status'] == 'skipped':
                    progress_counter['skipped'] += 1
                    print(f"⏭️  [{len(results)}/{progress_counter['total']}] {result['filename']} → {result['cloudinary_url']} (already uploaded)")
                else:
                    progress_counter['failed'] += 1
                    error_msg = f"Upload failed for {result['filename']}: {result.get('error', 'Unknown error')}"
                    logging.error(error_msg)
                    print(f"❌ [{len(results)}/{progress_counter['total']}] {error_msg}")
                    error_log.append(error_msg)
                    
            except Exception as e:
                img = future_to_image[future]
                completed_files.add(img['id'])  # Track even failed completions
                error_msg = f"Unexpected error processing {img['name']}: {e}"
                logging.error(error_msg)
                print(f"❌ {error_msg}")
                progress_counter['failed'] += 1
                results.append({
                    'local_filename': os.path.splitext(img['name'])[0],
                    'cloudinary_url': 'UPLOAD_FAILED',
                    'status': 'failed',
                    'error': str(e),
                    'filename': img['name']
                })
        
        # Verify we didn't miss any files
        if len(completed_files) != expected_total:
            missing_count = expected_total - len(completed_files)
            logging.warning(f"WARNING: {missing_count} files may have been missed in processing!")
            print(f"⚠️  WARNING: {missing_count} files may have been missed!")
    
    elapsed_time = time.time() - start_time
    
    # Save final progress
    save_progress(progress_counter, folder_id, folder_name, len(image_files), timestamp)
    
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
        # Add JPG conversion for all successful uploads
        for result in results:
            if result.get('status') == 'success' and result.get('cloudinary_url'):
                original_url = result['cloudinary_url']
                if is_cloudinary_url(original_url):
                    current_format = get_current_format(original_url)
                    if current_format == 'png':
                        result['jpg_url'] = convert_cloudinary_url_to_jpg(original_url)
                        print(f"🔄 Converted PNG to JPG: {result['local_filename']}")
                    else:
                        result['jpg_url'] = original_url  # Already JPG or other format
                else:
                    result['jpg_url'] = original_url  # Not a Cloudinary URL
            else:
                result['jpg_url'] = result.get('cloudinary_url', '')  # Failed uploads
        
        csv_columns = ['local_filename', 'cloudinary_url', 'jpg_url', 'folder_path']
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            
            # Count conversions for statistics
            png_conversions = sum(1 for result in results 
                                if result.get('status') == 'success' 
                                and result.get('cloudinary_url') != result.get('jpg_url'))
            
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
            logging.info(f"  PNG to JPG conversions: {png_conversions}")
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
            if png_conversions > 0:
                print(f"  🔄 PNG to JPG conversions: {png_conversions}")
                print(f"  💡 Use 'jpg_url' column for optimal JPG format")
            
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
        print("  list                                    : List YOUR own folders in Google Drive")
        print("  shared                                  : List files and folders shared with you (individual shares)")
        print("  drives                                  : List Shared Drives (Team Drives) and their folders")
        print("  cloudinary [--folder <name>]            : List all folders in Cloudinary or search for specific folder")
        print("  upload <folder_id_or_url> [options]     : Upload images from a Google Drive folder")
        print("\nExamples:")
        print("  python googledrive_tocloudinary.py list        # Show your own folders")
        print("  python googledrive_tocloudinary.py shared      # Show folders shared with you")
        print("  python googledrive_tocloudinary.py drives      # Show Shared Drives (Team Drives)")
        print("  python googledrive_tocloudinary.py cloudinary  # Show existing Cloudinary folders")
        print("  python googledrive_tocloudinary.py cloudinary --folder beauty  # Search for folders containing 'beauty'")
        print("  python googledrive_tocloudinary.py cloudinary --folder=products # Search for folders containing 'products'")
        print("  # Using folder ID:")
        print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE")
        print("  # Using Google Drive URL:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2?usp=sharing'")
        print("  # With custom folder name:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2' my_custom_folder")
        print("  # With custom settings:")
        print("  python googledrive_tocloudinary.py upload 'https://drive.google.com/drive/folders/1310NnlTK5tn8fX00TKF_BDAi0o7eK0d2' my_custom_folder 10 --no-recursive")
        print("\nUpload Arguments:")
        print("  folder_id_or_url : Google Drive folder ID OR full Google Drive URL")
        print("  destination_name : CUSTOM folder name for Cloudinary (optional, default: uses Drive folder name)")
        print("  threads          : Number of concurrent threads (optional, default: 5, optimized for SSL stability)")  
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
        print("  ✓ Use 'cloudinary' command to see existing folders before uploading")
        print("  ✓ Use 'cloudinary --folder <name>' to search for specific folders")
        print("\nCloudinary Command Options:")
        print("  cloudinary                    : List all folders in your Cloudinary account")
        print("  cloudinary --folder beauty    : Search for folders containing 'beauty' (case-insensitive)")
        print("  cloudinary --folder=products  : Alternative syntax for folder search")
        print("  ✓ Search is case-insensitive and matches partial names")
        print("  ✓ Useful for checking if a folder exists before uploading")
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
        
        print("📁 Scanning YOUR own folders in Google Drive...\n")
        folders = list_all_google_drive_folders(service)
        
        print(f"\n{'='*60}")
        print(f"📁 Your own folders found: {len(folders)}")
        print(f"{'='*60}")
        print("\n💡 Use 'shared' command to see files shared with you")
        print("💡 Use 'drives' command to see Shared Drives (Team Drives)")
        
    # Handle "shared" command to show files shared with me (not Shared Drives)
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
        
        print("📋 Scanning files and folders shared with you (not Shared Drives)...\n")
        shared_items = list_shared_with_me(service)
        
        print(f"\n{'='*60}")
        print(f"📁 Shared folders found: {len(shared_items['folders'])}")
        print(f"🖼️  Shared images found: {len(shared_items['files'])}")
        print(f"{'='*60}")
        print("\n💡 Use any folder ID above with the 'upload' command")
        print("💡 Use 'drives' command to see Shared Drives (Team Drives)")
        print("💡 Use 'list' command to see your own folders")
        
    # Handle "cloudinary" command to show existing folders in Cloudinary
    elif command == "cloudinary":
        print("Testing Cloudinary configuration...")
        is_configured, message = test_cloudinary_connection()
        print(f"  {message}")
        
        if not is_configured:
            print("\n⚠️  Please check your Cloudinary configuration in .env file")
            sys.exit(1)
        
        print("  ✓ Cloudinary verified\n")
        
        # Check for --folder argument
        search_term = None
        if len(sys.argv) > 2:
            args = sys.argv[2:]
            i = 0
            while i < len(args):
                if args[i] == '--folder' and i + 1 < len(args):
                    search_term = args[i + 1]
                    break
                elif args[i].startswith('--folder='):
                    search_term = args[i].split('=', 1)[1]
                    break
                i += 1
        
        if search_term:
            # Search for specific folder
            folders = search_cloudinary_folder(search_term)
        else:
            # List all folders
            folders = list_cloudinary_folders()
        
        if folders:
            print("\n💡 Use any existing folder name with the 'upload' command to add more images")
            print("💡 Use a new folder name with 'upload' command to create a new folder")
        else:
            if search_term:
                print("\n💡 No matching folders found - try a different search term")
            else:
                print("\n💡 Your Cloudinary account appears to be empty")
            print("💡 Use the 'upload' command to start uploading images from Google Drive")
        
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
        MAX_WORKERS = 5  # Increased default for better performance while maintaining SSL stability
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
        print("\nAvailable commands:")
        print("  list       : Show YOUR own folders in Google Drive")
        print("  shared     : Show files and folders shared with you")
        print("  drives     : Show Shared Drives (Team Drives)")
        print("  cloudinary : Show existing folders and images in Cloudinary")
        print("  upload     : Upload images from any folder (yours or shared)")
        print("\nRun 'python googledrive_tocloudinary.py' for detailed usage information")
        sys.exit(1)