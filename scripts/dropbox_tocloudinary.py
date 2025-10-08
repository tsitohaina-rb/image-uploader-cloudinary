import os
import csv
import sys
import json
import hashlib
import logging
import cloudinary.uploader
import dropbox
from pathlib import Path
from config import USE_FILENAME, UNIQUE_FILENAME
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import tempfile
from datetime import datetime

# Setup directory structure
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

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
        """Initialize cache for a specific Dropbox folder"""
        self.folder_path = folder_path
        
        # Create a unique cache file name based on the folder path
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()
        self.cache_file = os.path.join(CACHE_DIR, f'upload_cache_{folder_hash}.json')
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
    
    def is_uploaded(self, file_path: str) -> bool:
        """Check if a file was successfully uploaded in previous runs"""
        with self.lock:
            return file_path in self.cache['successful_uploads']
    
    def mark_uploaded(self, file_path: str, result: dict):
        """Mark a file as successfully uploaded"""
        with self.lock:
            self.cache['folder_path'] = self.folder_path
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['successful_uploads'][file_path] = {
                'timestamp': datetime.now().isoformat(),
                'cloudinary_url': result['cloudinary_url'],
                'public_id': result.get('public_id', ''),
            }
            if file_path in self.cache['failed_uploads']:
                del self.cache['failed_uploads'][file_path]
            self._save_cache()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed upload"""
        with self.lock:
            self.cache['folder_path'] = self.folder_path
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['failed_uploads'][file_path] = {
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

def upload_single_image_from_dropbox(dbx, dropbox_path, folder_name, cache):
    """
    Cloud-to-cloud: get a temporary Dropbox download URL and let Cloudinary fetch it directly.
    """
    logging.info(f"Processing: {dropbox_path}")
    
    # Already uploaded? return cached result and update progress
    if cache.is_uploaded(dropbox_path):
        with progress_lock:
            progress_counter['skipped'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SKIPPED: {dropbox_path} (previously uploaded)")
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        cached_data = cache.cache['successful_uploads'][dropbox_path]
        return {
            'local_filename': os.path.splitext(os.path.basename(dropbox_path))[0],
            'cloudinary_url': cached_data['cloudinary_url'],
            'status': 'skipped',
            'public_id': cached_data.get('public_id', '')
        }

    try:
        logging.info(f"START TRANSFER: {dropbox_path}")
        
        # 1) Ask Dropbox for a temporary direct link (valid ~4 hours)
        logging.info(f"  Getting Dropbox link: {dropbox_path}")
        tmp_link_resp = dbx.files_get_temporary_link(dropbox_path)
        tmp_url = tmp_link_resp.link  # public direct-download URL
        logging.info(f"  Got temporary link for: {dropbox_path}")

        # 2) Derive filename / extension for Cloudinary options
        filename = os.path.basename(dropbox_path)
        file_stem, ext = os.path.splitext(filename)
        original_extension = ext.lower().replace('.', '')
        logging.info(f"  Uploading to Cloudinary as: {file_stem}.{original_extension}")

        # 3) Cloudinary pulls the file from Dropbox URL (no local download)
        response = cloudinary.uploader.upload(
            tmp_url,
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
            'public_id': response.get('public_id', '')
        }

        # 4) Cache + progress
        cache.mark_uploaded(dropbox_path, result)
        with progress_lock:
            progress_counter['uploaded'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SUCCESS: {dropbox_path} → {result['cloudinary_url']}")
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
                error_log.append(f"{os.path.basename(dropbox_path)}: {error_message}")
            logging.error(f"FAILED: {dropbox_path}")
            logging.error(f"Error details: {error_message}")
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        file_stem = os.path.splitext(os.path.basename(dropbox_path))[0]
        cache.mark_failed(dropbox_path, error_message)
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


def test_dropbox_connection(token):
    """Test if Dropbox is properly configured."""
    try:
        if not token:
            return False, "Dropbox token is not configured"
        
        dbx = dropbox.Dropbox(token)
        # Test the connection
        dbx.users_get_current_account()
        return True, "Dropbox connection successful"
    except Exception as e:
        return False, f"Dropbox connection error: {str(e)}"


def list_all_dropbox_folders(dbx, folder_path='', indent=0):
    """
    Recursively list all folders in Dropbox.
    
    Args:
        dbx: Dropbox client instance
        folder_path (str): Starting folder path (empty string for root)
        indent (int): Indentation level for display
        
    Returns:
        list: List of folder paths
    """
    folders = []
    
    try:
        result = dbx.files_list_folder(folder_path)
        
        while True:
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FolderMetadata):
                    folder_display = entry.path_display
                    folders.append(folder_display)
                    print(f"{'  ' * indent}📁 {folder_display}")
                    
                    # Recursively list subfolders
                    subfolders = list_all_dropbox_folders(dbx, folder_display, indent + 1)
                    folders.extend(subfolders)
            
            if not result.has_more:
                break
            
            result = dbx.files_list_folder_continue(result.cursor)
        
    except Exception as e:
        print(f"Error accessing Dropbox folder '{folder_path}': {str(e)}")
    
    return folders


def get_images_from_dropbox_folder(dbx, folder_path):
    """
    Get list of image files from a Dropbox folder.
    
    Args:
        dbx: Dropbox client instance
        folder_path (str): Path to folder in Dropbox
        
    Returns:
        list: List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    image_files = []
    
    try:
        result = dbx.files_list_folder(folder_path)
        
        while True:
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    if os.path.splitext(entry.name)[1].lower() in image_extensions:
                        image_files.append(entry.path_display)
            
            if not result.has_more:
                break
            
            result = dbx.files_list_folder_continue(result.cursor)
        
    except Exception as e:
        print(f"Error accessing Dropbox folder '{folder_path}': {str(e)}")
    
    return image_files


def upload_dropbox_folder_to_cloudinary(dropbox_folder_path, max_workers=10):
    """
    Upload images from a Dropbox folder to Cloudinary using multi-threading.
    Supports resuming interrupted uploads through caching.
    
    Args:
        dropbox_folder_path (str): Path to folder in Dropbox (e.g., '/my_images')
        max_workers (int): Number of concurrent upload threads (default: 10)
    """
    
    # Get Dropbox token from environment
    DROPBOX_TOKEN = os.getenv('DROPBOX_TOKEN')
    
    if not DROPBOX_TOKEN:
        print("Error: DROPBOX_TOKEN not found in .env file")
        print("Please add DROPBOX_TOKEN=your_token_here to your .env file")
        return
    
    # Test connections
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print("\n⚠️  Please check your config.py and .env file")
        return
    
    print("  ✓ Cloudinary verified\n")
    
    print("Testing Dropbox connection...")
    is_connected, message = test_dropbox_connection(DROPBOX_TOKEN)
    print(f"  {message}")
    
    if not is_connected:
        print("\n⚠️  Please check your DROPBOX_TOKEN in .env file")
        print("Get your token from: https://www.dropbox.com/developers/apps")
        return
    
    print("  ✓ Dropbox verified\n")
    
    # Initialize Dropbox client
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    
    # Get folder name (last part of path)
    folder_name = os.path.basename(dropbox_folder_path.rstrip('/'))
    
    # Get all images from Dropbox folder
    print(f"Scanning Dropbox folder: {dropbox_folder_path}")
    image_files = get_images_from_dropbox_folder(dbx, dropbox_folder_path)
    
    if not image_files:
        print(f"No images found in '{dropbox_folder_path}'")
        return
    
    # Setup logging
    log_file = setup_logging(folder_name)
    
    # Generate CSV filename based on folder name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(OUTPUT_DIR, f"{folder_name}_{timestamp}.csv")
    
    # Initialize upload cache
    cache = UploadCache(dropbox_folder_path)
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
            executor.submit(upload_single_image_from_dropbox, dbx, img, folder_name, cache): img 
            for img in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                img = future_to_image[future]
                print(f"❌ Unexpected error processing {img}: {e}")
                results.append({
                    'local_filename': os.path.splitext(os.path.basename(img))[0],
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
    # Check if folder argument is provided
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [arguments]")
        print("\nCommands:")
        print("  list                     : List all folders in your Dropbox")
        print("  upload <folder> [threads]: Upload images from a Dropbox folder")
        print("\nExamples:")
        print("  python main.py list")
        print("  python main.py upload /my_images")
        print("  python main.py upload /photos/vacation 15")
        print("\nUpload Arguments:")
        print("  folder  : Path to the folder in Dropbox (must start with /)")
        print("  threads : Number of concurrent threads (optional, default: 10)")
        print("\nNote: Make sure DROPBOX_TOKEN is set in your .env file")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Handle "list" command to show all Dropbox folders
    if command == "list":
        DROPBOX_TOKEN = os.getenv('DROPBOX_TOKEN')
        
        if not DROPBOX_TOKEN:
            print("Error: DROPBOX_TOKEN not found in .env file")
            print("Please add DROPBOX_TOKEN=your_token_here to your .env file")
            sys.exit(1)
        
        print("Testing Dropbox connection...")
        is_connected, message = test_dropbox_connection(DROPBOX_TOKEN)
        print(f"  {message}")
        
        if not is_connected:
            print("\n⚠️  Please check your DROPBOX_TOKEN in .env file")
            print("Get your token from: https://www.dropbox.com/developers/apps")
            sys.exit(1)
        
        print("  ✓ Dropbox verified\n")
        
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        
        print("Scanning all folders in your Dropbox...\n")
        folders = list_all_dropbox_folders(dbx)
        
        print(f"\n{'='*60}")
        print(f"Total folders found: {len(folders)}")
        print(f"{'='*60}")
        
    # Handle "upload" command
    elif command == "upload":
        if len(sys.argv) < 3:
            print("Error: Please provide a folder path to upload")
            print("\nUsage: python main.py upload <dropbox_folder_path> [max_workers]")
            print("\nExamples:")
            print("  python main.py upload /my_images")
            print("  python main.py upload /photos/vacation 15")
            sys.exit(1)
        
        # Get folder path from command line argument
        DROPBOX_FOLDER = sys.argv[2]
        
        # Ensure path starts with /
        if not DROPBOX_FOLDER.startswith('/'):
            DROPBOX_FOLDER = '/' + DROPBOX_FOLDER
        
        # Get max_workers from command line argument (optional)
        MAX_WORKERS = 10  # Default value
        if len(sys.argv) >= 4:
            try:
                MAX_WORKERS = int(sys.argv[3])
            except ValueError:
                print(f"Warning: Invalid max_workers value '{sys.argv[3]}', using default: 10")
        
        upload_dropbox_folder_to_cloudinary(DROPBOX_FOLDER, max_workers=MAX_WORKERS)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("\nAvailable commands: list, upload")
        print("Run 'python main.py' for usage information")
        sys.exit(1)