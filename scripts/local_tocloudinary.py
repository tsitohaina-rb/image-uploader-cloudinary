import os
import csv
import sys
import json
import hashlib
import logging
import cloudinary.uploader
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import time
import urllib.parse
import re
from functools import wraps
from PIL import Image, ImageOps
from tqdm import tqdm
import platform
import glob

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import USE_FILENAME, UNIQUE_FILENAME
except ImportError:
    # If config import fails, use default values
    USE_FILENAME = True
    UNIQUE_FILENAME = False
from threading import Lock
from multiprocessing import Lock as MPLock, Manager
import time
from datetime import datetime

# Windows console encoding setup for emoji/Unicode support
if platform.system() == "Windows":
    try:
        # Try to enable UTF-8 mode on Windows (Python 3.7+)
        if hasattr(sys.stdout, 'reconfigure'):
            # Use getattr to avoid linting issues
            reconfigure = getattr(sys.stdout, 'reconfigure', None)
            if reconfigure:
                reconfigure(encoding='utf-8')
            reconfigure = getattr(sys.stderr, 'reconfigure', None)
            if reconfigure:
                reconfigure(encoding='utf-8')
        else:
            # Fallback for older Python versions
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')
    except (AttributeError, OSError):
        # Additional fallback if anything fails
        pass

# Cross-platform emoji/symbol compatibility
IS_WINDOWS = platform.system() == "Windows"

# Define symbols that work across platforms
SYMBOLS = {
    'check': 'OK' if IS_WINDOWS else '✓',
    'cross': 'X' if IS_WINDOWS else '❌',
    'warning': '[!]' if IS_WINDOWS else '⚠️',
    'upload': '[UP]' if IS_WINDOWS else '⬆️',
    'success': '[OK]' if IS_WINDOWS else '✓',
    'failed': '[FAIL]' if IS_WINDOWS else '❌',
    'skip': '[SKIP]' if IS_WINDOWS else '⏭️',
    'compress': '[ZIP]' if IS_WINDOWS else '🗜️',
    'folder': '[DIR]' if IS_WINDOWS else '📁',
}

# Setup directory structure
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Retry decorator for network operations
def retry_with_backoff(max_retries=3, backoff_factor=5, exceptions=(Exception,)):
    """
    Decorator to retry functions with exponential backoff on specified exceptions.
    Enhanced for SSL stability in multi-processing environments.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        backoff_factor (float): Multiplier for delay between retries
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
                    
                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def compress_image_for_cloudinary(image_data, filename, max_size_mb=20, quality=85):
    """
    Compress image data to reduce file size for Cloudinary upload.
    
    Args:
        image_data (bytes): Original image data
        filename (str): Original filename for logging
        max_size_mb (int): Maximum size in MB before compression
        quality (int): JPEG compression quality (1-100)
    
    Returns:
        tuple: (compressed_data, was_compressed, final_size_mb, compression_ratio)
    """
    try:
        # Check if compression is needed
        original_size = len(image_data)
        original_size_mb = original_size / (1024 * 1024)
        
        if original_size_mb <= max_size_mb:
            return image_data, False, original_size_mb, 1.0
        
        # Load image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert RGBA to RGB if necessary (for JPEG compatibility)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        
        # Apply exif orientation
        image = ImageOps.exif_transpose(image)
        
        # Compress image
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
        compressed_data = output_buffer.getvalue()
        
        # Calculate compression results
        final_size = len(compressed_data)
        final_size_mb = final_size / (1024 * 1024)
        compression_ratio = final_size / original_size
        
        logging.info(f"Compressed {filename}: {original_size_mb:.2f}MB → {final_size_mb:.2f}MB "
                    f"(ratio: {compression_ratio:.2f}, quality: {quality})")
        
        return compressed_data, True, final_size_mb, compression_ratio
        
    except Exception as e:
        logging.warning(f"Compression failed for {filename}: {e}")
        original_size = len(image_data)
        original_size_mb = original_size / (1024 * 1024)
        return image_data, False, original_size_mb, 1.0

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

def convert_cloudinary_url_to_format(url, target_format):
    """Convert a Cloudinary URL to specified format"""
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
            # Insert format transformation after 'upload'
            path_parts.insert(upload_index + 1, f'f_{target_format}')
            
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

def convert_cloudinary_url_to_jpg(url):
    """Convert a Cloudinary URL to JPG format (backward compatibility)"""
    return convert_cloudinary_url_to_format(url, 'jpg')

def sanitize_filename_for_cloudinary(filename):
    """
    Sanitize filename for Cloudinary upload by removing/replacing unsupported characters.
    
    Cloudinary public_id supports: a-z, A-Z, 0-9, -, _, /
    This function will:
    1. Remove file extension (handled separately)
    2. Replace spaces and common separators with underscores
    3. Remove or replace special characters
    4. Ensure the result is not empty
    
    Args:
        filename (str): Original filename (without extension)
        
    Returns:
        str: Sanitized filename safe for Cloudinary
    """
    if not filename:
        return "unnamed_file"
    
    # Replace common separators and spaces with underscores
    sanitized = filename.replace(' ', '_')
    sanitized = sanitized.replace('.', '_')
    sanitized = sanitized.replace(',', '_')
    sanitized = sanitized.replace(';', '_')
    sanitized = sanitized.replace(':', '_')
    sanitized = sanitized.replace('!', '')
    sanitized = sanitized.replace('?', '')
    sanitized = sanitized.replace('*', '')
    sanitized = sanitized.replace('+', '_')
    sanitized = sanitized.replace('=', '_')
    sanitized = sanitized.replace('@', '_at_')
    sanitized = sanitized.replace('&', '_and_')
    sanitized = sanitized.replace('#', '_hash_')
    sanitized = sanitized.replace('$', '_dollar_')
    sanitized = sanitized.replace('%', '_percent_')
    sanitized = sanitized.replace('^', '')
    sanitized = sanitized.replace('(', '')
    sanitized = sanitized.replace(')', '')
    sanitized = sanitized.replace('[', '')
    sanitized = sanitized.replace(']', '')
    sanitized = sanitized.replace('{', '')
    sanitized = sanitized.replace('}', '')
    sanitized = sanitized.replace('<', '')
    sanitized = sanitized.replace('>', '')
    sanitized = sanitized.replace('|', '_')
    sanitized = sanitized.replace('\\', '_')
    sanitized = sanitized.replace('"', '')
    sanitized = sanitized.replace("'", '')
    sanitized = sanitized.replace('`', '')
    sanitized = sanitized.replace('~', '')
    
    # Remove any remaining non-ASCII characters
    import unicodedata
    sanitized = unicodedata.normalize('NFKD', sanitized)
    sanitized = ''.join(c for c in sanitized if ord(c) < 128)
    
    # Keep only alphanumeric, hyphens, and underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', sanitized)
    
    # Clean up multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores and hyphens
    sanitized = sanitized.strip('_-')
    
    # Ensure we don't return empty string
    if not sanitized:
        return "unnamed_file"
    
    # Limit length to reasonable size (Cloudinary has limits)
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip('_-')
    
    return sanitized

# Configure logging
def setup_logging(folder_name):
    """Setup logging with timestamp and folder-specific log file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in folder_name)
    safe_folder_name = safe_folder_name.replace(' ', '_')
    log_filename = os.path.join(LOG_DIR, f'{safe_folder_name}_{timestamp}.log')
    
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

@retry_with_backoff(max_retries=3, backoff_factor=2)
def upload_worker(file_path, folder_name, cache_file, output_format='jpg', upload_mode='skip-existing'):
    """
    Worker function for ProcessPoolExecutor that uploads a single image.
    This function doesn't use shared locks to avoid pickle issues.
    """
    # Import cloudinary in worker process
    import cloudinary.uploader
    
    # Load cache for this worker (read-only check)
    cache_data = {}
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
    except Exception:
        cache_data = {'successful_uploads': {}, 'failed_uploads': {}}
    
    # Get file info for checks
    filename = os.path.basename(file_path)
    file_stem, ext = os.path.splitext(filename)
    
    # Sanitize the filename for Cloudinary
    safe_file_stem = sanitize_filename_for_cloudinary(file_stem)
    
    # Log filename sanitization if changed
    if safe_file_stem != file_stem:
        logging.info(f"Sanitized filename: '{file_stem}' → '{safe_file_stem}'")
    
    # Check local cache (only check - no Cloudinary API calls)
    if upload_mode != 'clear-cache' and file_path in cache_data.get('successful_uploads', {}):
        cached_data = cache_data['successful_uploads'][file_path]
        original_url = cached_data['cloudinary_url']
        
        # Generate format-specific URL for cached result
        format_url = original_url
        if is_cloudinary_url(original_url):
            current_format = get_current_format(original_url)
            if current_format and current_format.lower() != output_format.lower() and output_format != 'auto':
                format_url = convert_cloudinary_url_to_format(original_url, output_format)
        
        return {
            'local_filename': safe_file_stem,
            'cloudinary_url': cached_data['cloudinary_url'],
            f'{output_format}_url' if output_format != 'auto' else 'format_url': format_url,
            'status': 'skipped',
            'skip_reason': 'cache',
            'public_id': cached_data.get('public_id', ''),
            'file_path': file_path
        }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info (already have filename and file_stem from above)
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Determine overwrite behavior based on upload mode
        should_overwrite = upload_mode == 'overwrite'
        
        # Check if compression or format conversion is needed
        needs_compression = file_size_mb > 20.0
        needs_format_conversion = output_format != 'auto' and original_extension.lower() != output_format.lower()
        needs_processing = needs_compression or needs_format_conversion
        
        upload_source = file_path
        compressed_data = None
        target_format = output_format if output_format != 'auto' else original_extension
        
        if needs_processing:
            # Read and process the image (compress large files or convert format)
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            if needs_compression:
                # Compress large files (always convert to JPG for compression)
                compressed_data, was_compressed, final_size_mb, compression_ratio = compress_image_for_cloudinary(
                    original_data, filename
                )
                
                if was_compressed:
                    upload_source = io.BytesIO(compressed_data)
                    target_format = 'jpg'  # Compression always results in JPG
                else:
                    compressed_data = None  # Reset if compression failed
            elif needs_format_conversion:
                # Convert format without compression
                if target_format.lower() == 'jpg':
                    # Convert to JPG with high quality
                    compressed_data, was_compressed, final_size_mb, compression_ratio = compress_image_for_cloudinary(
                        original_data, filename, max_size_mb=999999, quality=95
                    )
                    if was_compressed:
                        upload_source = io.BytesIO(compressed_data)
                    else:
                        compressed_data = None
                # For other formats (PNG, WebP), let Cloudinary handle the conversion
        
        # Prepare upload parameters based on file characteristics
        if compressed_data or needs_format_conversion:
            upload_kwargs = {
                'folder': folder_name,
                'public_id': safe_file_stem,
                'format': target_format,
                'resource_type': 'image',
                'use_filename': True,
                'unique_filename': False,
                'overwrite': should_overwrite,
                'timeout': 120
            }
        else:
            upload_kwargs = {
                'folder': folder_name,
                'public_id': safe_file_stem,
                'format': original_extension if output_format == 'auto' else target_format,
                'resource_type': 'auto',
                'use_filename': True,
                'unique_filename': False,
                'overwrite': should_overwrite,
                'timeout': 60
            }
        
        # Simple upload without progress callback (to avoid serialization issues)
        response = cloudinary.uploader.upload(upload_source, **upload_kwargs)
        
        # Generate format-specific URL for successful upload
        original_url = response['secure_url']
        format_url = original_url
        if is_cloudinary_url(original_url):
            current_format = get_current_format(original_url)
            if current_format and current_format.lower() != output_format.lower() and output_format != 'auto':
                format_url = convert_cloudinary_url_to_format(original_url, output_format)
        
        result = {
            'local_filename': safe_file_stem,
            'cloudinary_url': response['secure_url'],
            f'{output_format}_url' if output_format != 'auto' else 'format_url': format_url,
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'file_size': file_size,
            'compressed': compressed_data is not None,
            'original_size_mb': file_size_mb,
            'final_size_mb': (len(compressed_data) if compressed_data else file_size) / (1024 * 1024),
            'file_path': file_path  # Include for cache update
        }
        
        return result
        
    except Exception as e:
        error_message = str(e)
        safe_file_stem = sanitize_filename_for_cloudinary(os.path.splitext(os.path.basename(file_path))[0])
        
        return {
            'local_filename': safe_file_stem,
            'cloudinary_url': 'UPLOAD_FAILED',
            f'{output_format}_url' if output_format != 'auto' else 'format_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message,
            'file_path': file_path  # Include for cache update
        }

class UploadCache:
    """Manages the cache of uploaded files to support resume functionality"""
    
    def __init__(self, source_path: str, folder_name: str):
        """Initialize cache for a specific local folder/file"""
        self.source_path = source_path
        self.folder_name = folder_name
        
        # Create cache file name based on folder name (more user-friendly)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in folder_name)
        safe_folder_name = safe_folder_name.replace(' ', '_')
        self.cache_file = os.path.join(CACHE_DIR, f'{safe_folder_name}_{timestamp}_cache.json')
        
        # Also check for existing cache files with same folder name (without timestamp)
        self.existing_cache_file = self._find_existing_cache_file(safe_folder_name)
        
        self.lock = Lock()
        self.cache = self._load_cache()
    
    def _find_existing_cache_file(self, safe_folder_name: str) -> str:
        """Find existing cache file for the same folder name"""
        cache_pattern = os.path.join(CACHE_DIR, f'{safe_folder_name}_*_cache.json')
        existing_files = glob.glob(cache_pattern)
        if existing_files:
            # Return the most recent cache file
            return max(existing_files, key=os.path.getmtime)
        return ""
    
    def _load_cache(self) -> dict:
        """Load the cache from file"""
        try:
            # First try to load from existing cache file if it exists
            cache_file_to_load = self.existing_cache_file if self.existing_cache_file else self.cache_file
            
            if os.path.exists(cache_file_to_load):
                with open(cache_file_to_load, 'r') as f:
                    cache_data = json.load(f)
                    
                # If we loaded from an existing file, copy it to our new timestamped file
                if self.existing_cache_file and cache_file_to_load != self.cache_file:
                    self.cache_file = cache_file_to_load  # Use the existing file
                    
                return cache_data
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

def upload_local_to_cloudinary(local_path, cloudinary_folder=None, max_workers=10, retry_mode='auto', output_format='jpg', upload_mode='skip-existing'):
    """
    Upload images from a local path (file or folder) to Cloudinary using multiprocessing.
    Supports resuming interrupted uploads through caching.
    
    Args:
        local_path (str): Path to local file or folder
        cloudinary_folder (str): Optional custom folder name for Cloudinary
        max_workers (int): Number of concurrent upload processes (default: 10)
        retry_mode (str): Retry mode - 'auto', 'true', 'false' (default: 'auto')
        output_format (str): Target output format - 'jpg', 'png', 'webp', 'auto' (default: 'jpg')
        upload_mode (str): Upload behavior - 'skip-existing', 'overwrite', 'clear-cache' (default: 'skip-existing')
    """
    
    # Test Cloudinary connection
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print(f"\n{SYMBOLS['warning']}  Please check your config.py and .env file")
        return
    
    print(f"  {SYMBOLS['check']} Cloudinary verified\n")
    
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
    safe_folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in cloudinary_folder)
    safe_folder_name = safe_folder_name.replace(' ', '_')
    output_csv = os.path.join(OUTPUT_DIR, f"{safe_folder_name}_{timestamp}.csv")
    
    # Initialize upload cache
    cache = UploadCache(local_path, cloudinary_folder)
    
    # Handle clear-cache mode
    if upload_mode == 'clear-cache':
        print(f"🗑️  Clearing cache for folder '{cloudinary_folder}'...")
        cache.cache = {
            'source_path': cache.source_path,
            'last_run': '',
            'successful_uploads': {},
            'failed_uploads': {}
        }
        cache._save_cache()
        print(f"  {SYMBOLS['check']} Cache cleared")
    
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
    logging.info(f"Upload mode: {upload_mode}")
    logging.info(f"Output format: {output_format}")
    logging.info(f"Found {len(image_files)} images to process")
    logging.info(f"Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    if cache_stats['successful'] > 0:
        logging.info(f"Cache found: {cache_stats['successful']} previously uploaded files will be skipped")
        logging.info(f"Last upload run: {cache_stats['last_run']}")
    logging.info(f"Using {max_workers} concurrent processes for optimized performance")
    
    # Log upload mode behavior
    if upload_mode == 'skip-existing':
        logging.info(f"Mode: Skip existing - will use cache to skip previously uploaded files")
    elif upload_mode == 'overwrite':
        logging.info(f"Mode: Overwrite - will overwrite existing files on Cloudinary (except cached)")
    elif upload_mode == 'clear-cache':
        logging.info(f"Mode: Clear cache - cleared cache, will upload/overwrite all files")
    
    logging.info(f"Features: Auto-compression for files >20MB, smart upload parameters, progress tracking")
    logging.info(f"Output will be saved to: {output_csv}")
    logging.info(f"Log file: {log_file}\n")
    
    start_time = time.time()
    results = []
    
    # Windows multiprocessing fix
    if platform.system() == "Windows":
        # Ensure proper multiprocessing context on Windows
        import multiprocessing
        if __name__ == '__main__':
            multiprocessing.freeze_support()
    
    # Use ProcessPoolExecutor for concurrent uploads
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks with cache file path instead of cache object
        future_to_image = {
            executor.submit(upload_worker, img, cloudinary_folder, cache.cache_file, output_format, upload_mode): img 
            for img in image_files
        }
        
        # Collect results as they complete and update cache in main process
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                results.append(result)
                
                # Update cache and progress in main process
                file_path = result.get('file_path', '')
                if result['status'] == 'success':
                    cache.mark_uploaded(file_path, result)
                    with progress_lock:
                        progress_counter['uploaded'] += 1
                        current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
                        logging.info(f"SUCCESS: {file_path} → {result['cloudinary_url']}")
                        if current % 10 == 0 or current == progress_counter['total']:
                            print(f"Progress: {current}/{progress_counter['total']} "
                                  f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                                  f"Skipped: {progress_counter['skipped']})")
                elif result['status'] == 'failed':
                    cache.mark_failed(file_path, result.get('error', 'Unknown error'))
                    with progress_lock:
                        progress_counter['failed'] += 1
                        if len(error_log) < 10:
                            error_log.append(f"{os.path.basename(file_path)}: {result.get('error', 'Unknown error')}")
                        logging.error(f"FAILED: {file_path}")
                        logging.error(f"Error details: {result.get('error', 'Unknown error')}")
                        current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
                        if current % 10 == 0 or current == progress_counter['total']:
                            print(f"Progress: {current}/{progress_counter['total']} "
                                  f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                                  f"Skipped: {progress_counter['skipped']})")
                elif result['status'] == 'skipped':
                    with progress_lock:
                        progress_counter['skipped'] += 1
                        current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
                        skip_reason = result.get('skip_reason', 'unknown')
                        if skip_reason == 'cache':
                            logging.info(f"SKIPPED: {file_path} (found in cache)")
                        else:
                            logging.info(f"SKIPPED: {file_path} (reason: {skip_reason})")
                        if current % 10 == 0 or current == progress_counter['total']:
                            print(f"Progress: {current}/{progress_counter['total']} "
                                  f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']}, "
                                  f"Skipped: {progress_counter['skipped']})")
                
            except Exception as e:
                img = future_to_image[future]
                print(f"{SYMBOLS['cross']} Unexpected error processing {img}: {e}")
                results.append({
                    'local_filename': os.path.splitext(os.path.basename(img))[0],
                    'cloudinary_url': 'UPLOAD_FAILED',
                    'status': 'failed',
                    'error': str(e)
                })
    
    elapsed_time = time.time() - start_time
    
    # Extract failed uploads for potential retry
    failed_uploads = [result for result in results if result.get('status') == 'failed']
    
    # Show failed uploads summary
    if failed_uploads:
        print(f"\n{SYMBOLS['cross']} FAILED UPLOADS ({len(failed_uploads)} files):")
        for i, failed in enumerate(failed_uploads[:10]):  # Show first 10
            filename = failed.get('local_filename', 'Unknown')
            error = failed.get('error', 'Unknown error')
            print(f"  {i+1}. {filename}: {error}")
        if len(failed_uploads) > 10:
            print(f"  ... and {len(failed_uploads) - 10} more")
        print("")
        
        # Determine if retry should happen based on retry_mode
        should_retry = False
        if retry_mode == 'true':
            should_retry = True
            print(f"\n{SYMBOLS['upload']} RETRY MODE: Enabled (--retry true)")
        elif retry_mode == 'false':
            should_retry = False
            print(f"\n{SYMBOLS['skip']} RETRY MODE: Disabled (--retry false) - Skipping retry")
        elif retry_mode == 'auto':
            # Auto mode: retry if failed count is less than 10% of total
            failure_rate = len(failed_uploads) / len(image_files) if len(image_files) > 0 else 0
            should_retry = failure_rate < 0.1  # Retry if less than 10% failed
            print(f"\n{SYMBOLS['upload']} RETRY MODE: Auto (failure rate: {failure_rate*100:.1f}%)")
            if should_retry:
                print(f"   {SYMBOLS['check']} Will retry (failure rate < 10%)")
            else:
                print(f"   {SYMBOLS['skip']} Skipping retry (failure rate >= 10%, manual review recommended)")
        
        if should_retry:
            print(f"\n{SYMBOLS['upload']} Retrying {len(failed_uploads)} failed uploads...")
            logging.info(f"RETRYING FAILED UPLOADS: {len(failed_uploads)} files")
            
            # Extract failed file paths for retry
            failed_file_paths = [failed.get('file_path', '') for failed in failed_uploads if failed.get('file_path')]
            
            if failed_file_paths:
                retry_start_time = time.time()
                retry_results = []
                
                # Reset progress counters for retry
                retry_progress = {'uploaded': 0, 'failed': 0, 'skipped': 0}
                
                # Use ProcessPoolExecutor for retry uploads
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit retry tasks
                    future_to_image = {
                        executor.submit(upload_worker, img, cloudinary_folder, cache.cache_file, output_format, upload_mode): img 
                        for img in failed_file_paths
                    }
                    
                    # Collect retry results
                    for future in as_completed(future_to_image):
                        try:
                            result = future.result()
                            retry_results.append(result)
                            
                            # Update cache and progress for retry
                            file_path = result.get('file_path', '')
                            if result['status'] == 'success':
                                cache.mark_uploaded(file_path, result)
                                retry_progress['uploaded'] += 1
                                logging.info(f"RETRY SUCCESS: {file_path} → {result['cloudinary_url']}")
                            elif result['status'] == 'failed':
                                cache.mark_failed(file_path, result.get('error', 'Unknown error'))
                                retry_progress['failed'] += 1
                                logging.error(f"RETRY FAILED: {file_path} - {result.get('error', 'Unknown error')}")
                            elif result['status'] == 'skipped':
                                retry_progress['skipped'] += 1
                                logging.info(f"RETRY SKIPPED: {file_path} (already uploaded)")
                            
                            # Update original results with retry outcome
                            for i, orig_result in enumerate(results):
                                if orig_result.get('file_path') == file_path:
                                    results[i] = result
                                    break
                        
                        except Exception as e:
                            img = future_to_image[future]
                            print(f"{SYMBOLS['cross']} Unexpected retry error processing {img}: {e}")
                            retry_progress['failed'] += 1
                
                retry_elapsed = time.time() - retry_start_time
                
                # Update final statistics
                successful = sum(1 for r in results if r['status'] == 'success')
                failed = sum(1 for r in results if r['status'] == 'failed')
                skipped = sum(1 for r in results if r['status'] == 'skipped')
                
                print(f"\n{SYMBOLS['upload']} RETRY COMPLETED in {retry_elapsed:.2f} seconds:")
                print(f"  Retry successful: {retry_progress['uploaded']}")
                print(f"  Retry failed: {retry_progress['failed']}")
                print(f"  Retry skipped: {retry_progress['skipped']}")
                
                # Show final statistics
                print(f"\n{SYMBOLS['check']} FINAL TOTALS:")
                print(f"  Success: {successful}")
                print(f"  Failed: {failed}")
                print(f"  Skipped: {skipped}")
                
                # Show remaining failed uploads if any
                remaining_failed = [result for result in results if result.get('status') == 'failed']
                if remaining_failed:
                    print(f"\n{SYMBOLS['cross']} STILL FAILED AFTER RETRY ({len(remaining_failed)} files):")
                    for failed in remaining_failed[:5]:  # Show first 5
                        filename = failed.get('local_filename', 'Unknown')
                        error = failed.get('error', 'Unknown error')
                        print(f"  - {filename}: {error}")
                    if len(remaining_failed) > 5:
                        print(f"  ... and {len(remaining_failed) - 5} more")
                else:
                    print(f"\n{SYMBOLS['success']} All uploads successful after retry!")
                
                logging.info(f"RETRY OPERATION COMPLETED:")
                logging.info(f"  Retry time: {retry_elapsed:.2f} seconds")
                logging.info(f"  Final totals - Success: {successful}, Failed: {failed}, Skipped: {skipped}")
            else:
                print(f"{SYMBOLS['cross']} Could not find original file paths for retry")
                logging.warning("Could not find original file paths for retry")
        else:
            if retry_mode == 'auto':
                print(f"{SYMBOLS['skip']} Skipping retry due to high failure rate")
            logging.info("Skipping retry based on retry mode setting")
    
    # Write results to CSV
    if results:
        # Dynamic column naming based on output format
        format_column_name = f'{output_format}_url' if output_format != 'auto' else 'format_url'
        csv_columns = ['local_filename', 'cloudinary_url', format_column_name]
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] == 'failed')
            skipped = sum(1 for r in results if r['status'] == 'skipped')
            compressed_count = sum(1 for r in results if r.get('compressed', False))
            
            print(f"\n{'='*60}")
            print(f"{SYMBOLS['check']} Upload completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                print(f"{SYMBOLS['check']} Average speed: {successful/elapsed_time:.2f} images/second")
            print(f"{SYMBOLS['check']} Results saved to '{output_csv}'")
            print(f"  Total images processed: {len(image_files)}")
            print(f"  Successfully uploaded: {successful}")
            print(f"  Previously uploaded (skipped): {skipped}")
            print(f"  Failed uploads: {failed}")
            print(f"  Auto-compressed files: {compressed_count}")
            print(f"  Total size processed: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
            
            # Cache statistics
            cache_stats = cache.get_stats()
            print(f"\nCache Status:")
            print(f"  Total files in cache: {cache_stats['successful']}")
            print(f"  Failed files in cache: {cache_stats['failed']}")
            print(f"  Last upload run: {cache_stats['last_run']}")
            
            if error_log:
                print(f"\n{SYMBOLS['warning']}  Sample errors (first 10):")
                for err in error_log:
                    print(f"  - {err}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error writing CSV: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Windows multiprocessing support
    if platform.system() == "Windows":
        import multiprocessing
        multiprocessing.freeze_support()
    
    # Check if path argument is provided
    if len(sys.argv) < 2:
        print("Usage: python local_tocloudinary.py <local_path> [cloudinary_folder] [max_workers] [options]")
        print("\nArguments:")
        print("  local_path        : Path to local file or folder containing images")
        print("  cloudinary_folder : Custom folder name for Cloudinary (optional)")
        print("  max_workers       : Number of concurrent processes (optional, default: 10)")
        print("\nOptions:")
        print("  --retry <mode>    : Retry mode for failed uploads (auto/true/false, default: auto)")
        print("                      auto: retry if failure rate < 10%")
        print("                      true: always retry failed uploads")
        print("                      false: never retry failed uploads")
        print("  --format <format> : Output format for uploaded images (jpg/png/webp/auto, default: jpg)")
        print("                      jpg: convert all images to JPG format")
        print("                      png: convert all images to PNG format")
        print("                      webp: convert all images to WebP format")
        print("                      auto: keep original format")
        print("  --overwrite       : Overwrite existing files on Cloudinary")
        print("  --skip-existing   : Skip files that already exist in cache (default)")
        print("  --clear-cache     : Clear local cache and re-upload everything")
        print("\nUpload Modes:")
        print("  --skip-existing   : Use cache to skip previously uploaded files (DEFAULT)")
        print("  --overwrite       : Use cache but overwrite existing files on Cloudinary")
        print("  --clear-cache     : Clear cache and upload/overwrite everything")
        print("\nFeatures:")
        print("  • Auto-compression for files >20MB to optimize upload speed")
        print("  • Smart upload parameters based on file type and compression")
        print("  • Progress tracking with individual file progress bars")
        print("  • Retry mechanism with exponential backoff for reliability")
        print("  • ProcessPoolExecutor for optimal CPU utilization")
        print("  • Resume capability with upload caching")
        print("  • Cross-platform compatibility (Windows, macOS, Linux)")
        print("  • Flexible output format conversion (JPG, PNG, WebP)")
        print("  • Intelligent duplicate detection (cache-based)")
        print("\nExamples:")
        print("  python local_tocloudinary.py /path/to/images")
        print("  python local_tocloudinary.py /path/to/images my_photos")
        print("  python local_tocloudinary.py /path/to/images --overwrite")
        print("  python local_tocloudinary.py /path/to/images --clear-cache")
        print("  python local_tocloudinary.py /path/to/images --format png")
        print("  python local_tocloudinary.py /path/to/images my_photos 15 --retry false --format webp --overwrite")
        print("\nSupported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .tiff, .tif")
        print("\nNote: Make sure Cloudinary credentials are configured in config.py and .env")
        sys.exit(1)
    
    # Get local path from command line argument
    LOCAL_PATH = sys.argv[1]
    
    # Initialize defaults
    CLOUDINARY_FOLDER = None
    MAX_WORKERS = 10
    RETRY_MODE = 'auto'
    OUTPUT_FORMAT = 'jpg'
    UPLOAD_MODE = 'skip-existing'  # Default mode
    
    # Parse arguments
    args = sys.argv[2:]  # Skip script name and local path
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        if arg == '--retry':
            if i + 1 < len(args):
                retry_value = args[i + 1].lower()
                if retry_value in ['auto', 'true', 'false']:
                    RETRY_MODE = retry_value
                    i += 2  # Skip both --retry and its value
                    continue
                else:
                    print(f"Warning: Invalid retry mode '{args[i + 1]}', using default (auto)")
                    i += 2
                    continue
            else:
                print("Warning: --retry requires a value (auto/true/false), using default (auto)")
                i += 1
                continue
        elif arg == '--format':
            if i + 1 < len(args):
                format_value = args[i + 1].lower()
                if format_value in ['jpg', 'jpeg', 'png', 'webp', 'auto']:
                    OUTPUT_FORMAT = 'jpg' if format_value == 'jpeg' else format_value
                    i += 2  # Skip both --format and its value
                    continue
                else:
                    print(f"Warning: Invalid format '{args[i + 1]}', using default (jpg)")
                    i += 2
                    continue
            else:
                print("Warning: --format requires a value (jpg/png/webp/auto), using default (jpg)")
                i += 1
                continue
        elif arg == '--overwrite':
            UPLOAD_MODE = 'overwrite'
            i += 1
            continue
        elif arg == '--skip-existing':
            UPLOAD_MODE = 'skip-existing'
            i += 1
            continue
        elif arg == '--clear-cache':
            UPLOAD_MODE = 'clear-cache'
            i += 1
            continue
        
        # If not a flag, check if it's a positional argument
        if arg.isdigit():
            MAX_WORKERS = int(arg)
        elif not arg.startswith('--'):
            if CLOUDINARY_FOLDER is None:
                CLOUDINARY_FOLDER = arg
            elif arg.isdigit():
                MAX_WORKERS = int(arg)
        
        i += 1
    
    print(f"📁 Local path: {LOCAL_PATH}")
    if CLOUDINARY_FOLDER:
        print(f"☁️  Cloudinary folder: {CLOUDINARY_FOLDER}")
    print(f"👥 Max workers: {MAX_WORKERS}")
    print(f"🔄 Retry mode: {RETRY_MODE}")
    print(f"🎨 Output format: {OUTPUT_FORMAT}")
    print(f"📋 Upload mode: {UPLOAD_MODE}")
    print()
    
    upload_local_to_cloudinary(LOCAL_PATH, CLOUDINARY_FOLDER, max_workers=MAX_WORKERS, retry_mode=RETRY_MODE, output_format=OUTPUT_FORMAT, upload_mode=UPLOAD_MODE)