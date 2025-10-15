import os
import csv
import sys
import json
import hashlib
import logging
import cloudinary.uploader
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pathlib import Path
from config import USE_FILENAME, UNIQUE_FILENAME
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

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

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

def upload_single_image_from_gdrive(service, file_info, folder_name, cache):
    """
    Cloud-to-cloud: get a temporary Google Drive download URL and let Cloudinary fetch it directly.
    """
    file_id = file_info['id']
    filename = file_info['name']
    
    logging.info(f"Processing: {filename} (ID: {file_id})")
    
    # Already uploaded? return cached result and update progress
    if cache.is_uploaded(file_id):
        with progress_lock:
            progress_counter['skipped'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SKIPPED: {filename} (previously uploaded)")
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        cached_data = cache.cache['successful_uploads'][file_id]
        return {
            'local_filename': os.path.splitext(filename)[0],
            'cloudinary_url': cached_data['cloudinary_url'],
            'status': 'skipped',
            'public_id': cached_data.get('public_id', '')
        }

    try:
        logging.info(f"START TRANSFER: {filename}")
        
        # 1) Get Google Drive download URL
        logging.info(f"  Getting Google Drive download URL: {filename}")
        
        # For Google Drive, we need to construct the download URL
        # Public files can use: https://drive.google.com/uc?id=FILE_ID&export=download
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        logging.info(f"  Got download URL for: {filename}")

        # 2) Derive filename / extension for Cloudinary options
        file_stem, ext = os.path.splitext(filename)
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        logging.info(f"  Uploading to Cloudinary as: {file_stem}.{original_extension}")

        # 3) Cloudinary pulls the file from Google Drive URL (no local download)
        response = cloudinary.uploader.upload(
            download_url,
            folder=folder_name,
            public_id=file_stem,       # use original filename without extension
            use_filename=False,        # don't use remote URL name
            unique_filename=False,     # keep stable public_id
            overwrite=True,            # allow re-runs to overwrite
            format=original_extension, # keep original extension
            resource_type="image"      # or "auto" if you might have videos/svg/others
        )

        result = {
            'local_filename': file_stem,
            'cloudinary_url': response['secure_url'],
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'filename': filename
        }

        # 4) Cache + progress
        cache.mark_uploaded(file_id, result)
        with progress_lock:
            progress_counter['uploaded'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SUCCESS: {filename} → {result['cloudinary_url']}")
            if current % 100 == 0 or current == progress_counter['total']:
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
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        file_stem = os.path.splitext(filename)[0]
        cache.mark_failed(file_id, error_message)
        return {
            'local_filename': file_stem,
            'cloudinary_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message
        }

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

def get_images_from_gdrive_folder(service, folder_id):
    """
    Get list of image files from a Google Drive folder.
    
    Args:
        service: Google Drive service instance
        folder_id (str): Folder ID in Google Drive
        
    Returns:
        list: List of image file information dictionaries
    """
    image_mimetypes = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
        'image/bmp', 'image/webp', 'image/svg+xml'
    }
    image_files = []
    
    try:
        # Build query for image files in the folder
        mimetype_query = " or ".join([f"mimeType='{mt}'" for mt in image_mimetypes])
        query = f"'{folder_id}' in parents and ({mimetype_query}) and trashed=false"
        
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token
            ).execute()
            
            items = results.get('files', [])
            for item in items:
                image_files.append({
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'size': item.get('size', 0)
                })
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{folder_id}': {str(e)}")
    
    return image_files

def upload_gdrive_folder_to_cloudinary(folder_id, folder_name=None, max_workers=10):
    """
    Upload images from a Google Drive folder to Cloudinary using multi-threading.
    Supports resuming interrupted uploads through caching.
    
    Args:
        folder_id (str): Google Drive folder ID
        folder_name (str): Optional custom folder name for Cloudinary (default: uses Drive folder name)
        max_workers (int): Number of concurrent upload threads (default: 10)
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
    
    # Get all images from Google Drive folder
    print(f"Scanning Google Drive folder: {folder_name} (ID: {folder_id})")
    image_files = get_images_from_gdrive_folder(service, folder_id)
    
    if not image_files:
        print(f"No images found in folder '{folder_name}'")
        return
    
    # Setup logging
    log_file = setup_logging(folder_name)
    
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
    logging.info(f"Found {len(image_files)} images to process")
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
        # Submit all upload tasks
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
                    'error': str(e)
                })
    
    elapsed_time = time.time() - start_time
    
    # Write results to CSV
    if results:
        csv_columns = ['local_filename', 'cloudinary_url']
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] == 'failed')
            skipped = sum(1 for r in results if r['status'] == 'skipped')
            
            print(f"\n{'='*60}")
            print(f"✓ Upload completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                print(f"✓ Average speed: {successful/elapsed_time:.2f} images/second")
            print(f"✓ Results saved to '{output_csv}'")
            print(f"  Total images processed: {len(image_files)}")
            print(f"  Successfully uploaded: {successful}")
            print(f"  Previously uploaded (skipped): {skipped}")
            print(f"  Failed uploads: {failed}")
            
            # Cache statistics
            cache_stats = cache.get_stats()
            print(f"\nCache Status:")
            print(f"  Total files in cache: {cache_stats['successful']}")
            print(f"  Failed files in cache: {cache_stats['failed']}")
            print(f"  Last upload run: {cache_stats['last_run']}")
            
            if error_log:
                print(f"\n⚠️  Sample errors (first 10):")
                for err in error_log:
                    print(f"  - {err}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error writing CSV: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Check if command argument is provided
    if len(sys.argv) < 2:
        print("Usage: python googledrive_tocloudinary.py <command> [arguments]")
        print("\nCommands:")
        print("  list                           : List all folders in your Google Drive")
        print("  upload <folder_id> [name] [threads] : Upload images from a Google Drive folder")
        print("\nExamples:")
        print("  python googledrive_tocloudinary.py list")
        print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE")
        print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE my_photos 15")
        print("\nUpload Arguments:")
        print("  folder_id : Google Drive folder ID (from the URL or 'list' command)")
        print("  name      : Custom folder name for Cloudinary (optional)")
        print("  threads   : Number of concurrent threads (optional, default: 10)")
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
        
    # Handle "upload" command
    elif command == "upload":
        if len(sys.argv) < 3:
            print("Error: Please provide a Google Drive folder ID")
            print("\nUsage: python googledrive_tocloudinary.py upload <folder_id> [folder_name] [max_workers]")
            print("\nExamples:")
            print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE")
            print("  python googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE my_photos 15")
            sys.exit(1)
        
        # Get folder ID from command line argument
        FOLDER_ID = sys.argv[2]
        
        # Get custom folder name (optional)
        FOLDER_NAME = None
        if len(sys.argv) >= 4 and not sys.argv[3].isdigit():
            FOLDER_NAME = sys.argv[3]
        
        # Get max_workers from command line argument (optional)
        MAX_WORKERS = 10  # Default value
        if len(sys.argv) >= 4:
            try:
                # If third argument is a number, it's max_workers
                if sys.argv[3].isdigit():
                    MAX_WORKERS = int(sys.argv[3])
                # If fourth argument exists and is a number, it's max_workers
                elif len(sys.argv) >= 5 and sys.argv[4].isdigit():
                    MAX_WORKERS = int(sys.argv[4])
            except ValueError:
                print(f"Warning: Invalid max_workers value, using default: 10")
        
        upload_gdrive_folder_to_cloudinary(FOLDER_ID, FOLDER_NAME, max_workers=MAX_WORKERS)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("\nAvailable commands: list, upload")
        print("Run 'python googledrive_tocloudinary.py' for usage information")
        sys.exit(1)