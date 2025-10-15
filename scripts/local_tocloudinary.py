import os
import csv
import sys
import json
import hashlib
import logging
import cloudinary.uploader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import USE_FILENAME, UNIQUE_FILENAME
except ImportError:
    # If config import fails, use default values
    USE_FILENAME = True
    UNIQUE_FILENAME = False
from threading import Lock
import time
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
    
    def __init__(self, source_path: str):
        """Initialize cache for a specific local folder/file"""
        self.source_path = source_path
        
        # Create a unique cache file name based on the source path
        source_hash = hashlib.md5(source_path.encode()).hexdigest()
        self.cache_file = os.path.join(CACHE_DIR, f'local_upload_cache_{source_hash}.json')
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
            'source_path': '',
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
            self.cache['source_path'] = self.source_path
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['successful_uploads'][file_path] = {
                'timestamp': datetime.now().isoformat(),
                'cloudinary_url': result['cloudinary_url'],
                'public_id': result.get('public_id', ''),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            if file_path in self.cache['failed_uploads']:
                del self.cache['failed_uploads'][file_path]
            self._save_cache()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed upload"""
        with self.lock:
            self.cache['source_path'] = self.source_path
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

def upload_single_image_from_local(file_path, folder_name, cache):
    """
    Upload a single local image file to Cloudinary.
    """
    logging.info(f"Processing: {file_path}")
    
    # Already uploaded? return cached result and update progress
    if cache.is_uploaded(file_path):
        with progress_lock:
            progress_counter['skipped'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SKIPPED: {file_path} (previously uploaded)")
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        cached_data = cache.cache['successful_uploads'][file_path]
        return {
            'local_filename': os.path.splitext(os.path.basename(file_path))[0],
            'cloudinary_url': cached_data['cloudinary_url'],
            'status': 'skipped',
            'public_id': cached_data.get('public_id', '')
        }

    try:
        logging.info(f"START UPLOAD: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        filename = os.path.basename(file_path)
        file_stem, ext = os.path.splitext(filename)
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        file_size = os.path.getsize(file_path)
        
        logging.info(f"  File size: {file_size:,} bytes")
        logging.info(f"  Uploading to Cloudinary as: {file_stem}.{original_extension}")

        # Upload to Cloudinary
        response = cloudinary.uploader.upload(
            file_path,
            folder=folder_name,
            public_id=file_stem,       # use original filename without extension
            use_filename=True,         # use filename for public_id
            unique_filename=False,     # keep stable public_id
            overwrite=True,            # allow re-runs to overwrite
            resource_type="auto"       # auto-detect resource type
        )

        result = {
            'local_filename': file_stem,
            'cloudinary_url': response['secure_url'],
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'file_size': file_size
        }

        # Cache + progress
        cache.mark_uploaded(file_path, result)
        with progress_lock:
            progress_counter['uploaded'] += 1
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            logging.info(f"SUCCESS: {file_path} → {result['cloudinary_url']}")
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
                error_log.append(f"{os.path.basename(file_path)}: {error_message}")
            logging.error(f"FAILED: {file_path}")
            logging.error(f"Error details: {error_message}")
            current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                      f"Skipped: {progress_counter['skipped']})")

        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        cache.mark_failed(file_path, error_message)
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

def get_image_files_from_path(path):
    """
    Get list of image files from a local path (file or directory).
    
    Args:
        path (str): Path to file or directory
        
    Returns:
        list: List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.tif'}
    image_files = []
    
    path = Path(path)
    
    if path.is_file():
        # Single file
        if path.suffix.lower() in image_extensions:
            image_files.append(str(path.absolute()))
        else:
            print(f"Warning: {path} is not a supported image format")
    elif path.is_dir():
        # Directory - recursively find all image files
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path.absolute()))
    else:
        print(f"Error: Path '{path}' does not exist")
    
    return sorted(image_files)

def upload_local_to_cloudinary(local_path, cloudinary_folder=None, max_workers=10):
    """
    Upload images from a local path (file or folder) to Cloudinary using multi-threading.
    Supports resuming interrupted uploads through caching.
    
    Args:
        local_path (str): Path to local file or folder
        cloudinary_folder (str): Optional custom folder name for Cloudinary
        max_workers (int): Number of concurrent upload threads (default: 10)
    """
    
    # Test Cloudinary connection
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print("\n⚠️  Please check your config.py and .env file")
        return
    
    print("  ✓ Cloudinary verified\n")
    
    # Validate local path
    if not os.path.exists(local_path):
        print(f"Error: Path '{local_path}' does not exist")
        return
    
    # Determine folder name for Cloudinary
    if not cloudinary_folder:
        if os.path.isfile(local_path):
            # For single file, use parent directory name
            cloudinary_folder = os.path.basename(os.path.dirname(local_path))
        else:
            # For directory, use directory name
            cloudinary_folder = os.path.basename(local_path.rstrip('/\\'))
    
    # Get all image files
    print(f"Scanning local path: {local_path}")
    image_files = get_image_files_from_path(local_path)
    
    if not image_files:
        print(f"No images found in '{local_path}'")
        return
    
    # Setup logging
    log_file = setup_logging(cloudinary_folder)
    
    # Generate CSV filename based on folder name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(OUTPUT_DIR, f"{cloudinary_folder}_{timestamp}.csv")
    
    # Initialize upload cache
    cache = UploadCache(local_path)
    cache_stats = cache.get_stats()
    
    # Initialize progress counter
    progress_counter['total'] = len(image_files)
    progress_counter['uploaded'] = 0
    progress_counter['failed'] = 0
    progress_counter['skipped'] = 0
    error_log.clear()
    
    # Calculate total file size
    total_size = sum(os.path.getsize(f) for f in image_files if os.path.exists(f))
    
    logging.info(f"Processing local path: {local_path}")
    logging.info(f"Cloudinary folder: {cloudinary_folder}")
    logging.info(f"Found {len(image_files)} images to process")
    logging.info(f"Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
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
            executor.submit(upload_single_image_from_local, img, cloudinary_folder, cache): img 
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
            print(f"  Total size processed: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
            
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
    # Check if path argument is provided
    if len(sys.argv) < 2:
        print("Usage: python local_tocloudinary.py <local_path> [cloudinary_folder] [max_workers]")
        print("\nArguments:")
        print("  local_path        : Path to local file or folder containing images")
        print("  cloudinary_folder : Custom folder name for Cloudinary (optional)")
        print("  max_workers       : Number of concurrent threads (optional, default: 10)")
        print("\nExamples:")
        print("  python local_tocloudinary.py /path/to/images")
        print("  python local_tocloudinary.py /path/to/images my_photos")
        print("  python local_tocloudinary.py /path/to/single_image.jpg")
        print("  python local_tocloudinary.py /path/to/images my_photos 15")
        print("\nSupported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .tiff, .tif")
        print("\nNote: Make sure Cloudinary credentials are configured in config.py and .env")
        sys.exit(1)
    
    # Get local path from command line argument
    LOCAL_PATH = sys.argv[1]
    
    # Get custom cloudinary folder name (optional)
    CLOUDINARY_FOLDER = None
    if len(sys.argv) >= 3 and not sys.argv[2].isdigit():
        CLOUDINARY_FOLDER = sys.argv[2]
    
    # Get max_workers from command line argument (optional)
    MAX_WORKERS = 10  # Default value
    if len(sys.argv) >= 3:
        try:
            # If second argument is a number, it's max_workers
            if sys.argv[2].isdigit():
                MAX_WORKERS = int(sys.argv[2])
            # If third argument exists and is a number, it's max_workers
            elif len(sys.argv) >= 4 and sys.argv[3].isdigit():
                MAX_WORKERS = int(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid max_workers value, using default: 10")
    
    upload_local_to_cloudinary(LOCAL_PATH, CLOUDINARY_FOLDER, max_workers=MAX_WORKERS)