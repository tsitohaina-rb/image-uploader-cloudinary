""" 
Cloud Storage to Cloudinary Upload Script - Unified Multiprocessing Version
============================================================================

Cross-platform compatible script for uploading images from Google Drive or Dropbox to Cloudinary.
Combines the best features from both googledrive_tocloudinary.py and dropbox_tocloudinary.py.
Supports Windows, Linux, and macOS with:
- Fast multiprocessing for maximum performance (Google Drive architecture)
- Support for both Google Drive and Dropbox
- Cross-platform file paths with pathlib
- Platform-specific file locking (fcntl for Unix, msvcrt for Windows)
- Cloudinary folder management with user prompts
- Comprehensive caching and resume capabilities
- Unified command-line interface for both providers

Author: Tsitohaina
Date: November 2024
Modified: Combined Google Drive and Dropbox functionality
"""

# Suppress deprecation warnings from dropbox library
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dropbox.session")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import csv
import ssl
import sys
import json
import glob
import time
import socket
import hashlib
import logging
import platform
import requests
import unicodedata
import urllib.parse
import cloudinary
import cloudinary.uploader
import cloudinary.api
import re
import tempfile
import io
import pickle
from tqdm import tqdm
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageOps
import traceback

# Cloud storage imports
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import dropbox

from pathlib import Path
from multiprocessing import Manager, Lock as MPLock
from dotenv import load_dotenv
import multiprocessing

# Windows console encoding setup for emoji/Unicode support
if platform.system() == "Windows":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# Cross-platform emoji/symbol compatibility
IS_WINDOWS = platform.system() == "Windows"

# Define symbols that work across platforms
SYMBOLS = {
    'check': 'OK' if IS_WINDOWS else '✓',
    'cross': 'X' if IS_WINDOWS else '❌',
    'folder': '[DIR]' if IS_WINDOWS else '📁',
    'file': '[FILE]' if IS_WINDOWS else '📄',
    'image': '[IMG]' if IS_WINDOWS else '🖼️',
    'search': '[SEARCH]' if IS_WINDOWS else '🔍',
    'lock': '[LOCK]' if IS_WINDOWS else '🔐',
    'chart': '[STATS]' if IS_WINDOWS else '📊',
    'warning': '[!]' if IS_WINDOWS else '⚠️',
    'info': '[i]' if IS_WINDOWS else '💡',
    'save': '[SAVE]' if IS_WINDOWS else '💾',
    'reload': '[RELOAD]' if IS_WINDOWS else '🔄',
    'new': '[NEW]' if IS_WINDOWS else '🆕',
    'skip': 'SKIP' if IS_WINDOWS else '⏭',
    'compress': '[ZIP]' if IS_WINDOWS else '🗜️',
    'party': '[DONE]' if IS_WINDOWS else '🎉',
    'target': '[FIND]' if IS_WINDOWS else '🎯',
    'empty': '[EMPTY]' if IS_WINDOWS else '📭',
    'list': '[LIST]' if IS_WINDOWS else '📋',
    'drive': '[DRIVE]' if IS_WINDOWS else '🚗',
    'dropbox': '[BOX]' if IS_WINDOWS else '📦',
    'thinking': '[?]' if IS_WINDOWS else '🤔',
    'edit': '[EDIT]' if IS_WINDOWS else '📝',
}

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
        try:
            import fcntl
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        except (OSError, IOError, ImportError):
            pass
    elif HAS_FCNTL is False:
        try:
            import msvcrt
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
        except (OSError, IOError, ImportError, AttributeError):
            pass

def unlock_file(file_handle):
    """Cross-platform file unlocking"""
    if HAS_FCNTL:
        try:
            import fcntl
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError, ImportError):
            pass
    elif HAS_FCNTL is False:
        try:
            import msvcrt
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (OSError, IOError, ImportError, AttributeError):
            pass

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

# Google Drive API scopes
GDRIVE_SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly'
]

# Cloudinary utility functions
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

def convert_cloudinary_url_with_quality(url, target_format, quality=None):
    """
    Convert a Cloudinary URL to the specified format and quality using f_format transformation.
    
    Args:
        url (str): Original Cloudinary URL
        target_format (str): Target format (jpg, png, webp, etc.)
        quality (int, optional): Quality setting for lossy formats
    
    Returns:
        str: Converted URL with f_format transformation
    """
    if not is_cloudinary_url(url):
        return url  # Return as-is if not a Cloudinary URL
    
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
            # Build transformation string
            transformations = []
            
            # Add format transformation
            transformations.append(f'f_{target_format}')
            
            # Add quality if specified and format supports it
            if quality and target_format.lower() in ['jpg', 'jpeg', 'webp']:
                transformations.append(f'q_{quality}')
            
            # Insert transformations after 'upload'
            if transformations:
                transformation_string = ','.join(transformations)
                path_parts.insert(upload_index + 1, transformation_string)
            
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

def compress_image_for_cloudinary(image_data, filename, max_size_mb=19, quality_start=85):
    """
    Compress image data to be under Cloudinary's 20MB limit with a small safety margin.
    
    Args:
        image_data (bytes): Original image data
        filename (str): Filename for logging
        max_size_mb (int): Maximum size in MB (default: 19MB for small safety margin)
        quality_start (int): Starting JPEG quality (default: 85)
        
    Returns:
        tuple: (compressed_data, was_compressed, final_size_mb, compression_ratio)
    """
    try:
        original_size = len(image_data)
        original_size_mb = original_size / (1024 * 1024)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # If already under limit, return as-is
        if original_size <= max_size_bytes:
            logging.info(f"  Image {filename} is {original_size_mb:.2f} MB - no compression needed")
            return image_data, False, original_size_mb, 1.0
        
        logging.info(f"  Compressing {filename}: {original_size_mb:.2f} MB -> target: <{max_size_mb} MB")
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (for JPEG)
        if image.mode in ('RGBA', 'P', 'LA'):
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply orientation from EXIF if present
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass  # Continue if EXIF processing fails
        
        # Progressive compression with quality reduction
        current_quality = quality_start
        min_quality = 30  # Minimum acceptable quality
        quality_step = 10
        
        best_data = None
        best_size = float('inf')
        
        while current_quality >= min_quality:
            # Create compressed version
            output = io.BytesIO()
            
            # Save as progressive JPEG for better compression
            image.save(output, 
                      format='JPEG',
                      quality=current_quality,
                      optimize=True,
                      progressive=True)
            
            compressed_data = output.getvalue()
            compressed_size = len(compressed_data)
            
            if compressed_size <= max_size_bytes:
                # Found acceptable compression
                final_size_mb = compressed_size / (1024 * 1024)
                compression_ratio = compressed_size / original_size
                
                logging.info(f"  Compression successful: {filename}")
                logging.info(f"    Original: {original_size_mb:.2f} MB")
                logging.info(f"    Compressed: {final_size_mb:.2f} MB (quality: {current_quality})")
                logging.info(f"    Compression ratio: {compression_ratio:.2f} ({(1-compression_ratio)*100:.1f}% reduction)")
                
                return compressed_data, True, final_size_mb, compression_ratio
            
            # Track best result so far
            if compressed_size < best_size:
                best_data = compressed_data
                best_size = compressed_size
            
            current_quality -= quality_step
        
        # If we couldn't get under the limit, try dimension reduction
        if best_size > max_size_bytes:
            logging.warning(f"  Quality reduction insufficient for {filename}, trying dimension reduction...")
            
            # Try reducing dimensions by 20% each iteration
            scale_factor = 0.8
            current_image = image.copy()
            
            for attempt in range(3):  # Max 3 attempts at dimension reduction
                new_width = int(current_image.width * scale_factor)
                new_height = int(current_image.height * scale_factor)
                
                if new_width < 100 or new_height < 100:
                    break  # Don't make image too small
                
                resized_image = current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                output = io.BytesIO()
                resized_image.save(output,
                                 format='JPEG',
                                 quality=70,  # Use reasonable quality for resized image
                                 optimize=True,
                                 progressive=True)
                
                compressed_data = output.getvalue()
                compressed_size = len(compressed_data)
                
                if compressed_size <= max_size_bytes:
                    final_size_mb = compressed_size / (1024 * 1024)
                    compression_ratio = compressed_size / original_size
                    
                    logging.info(f"  Compression with resizing successful: {filename}")
                    logging.info(f"    Original: {original_size_mb:.2f} MB ({image.width}x{image.height})")
                    logging.info(f"    Compressed: {final_size_mb:.2f} MB ({new_width}x{new_height}, quality: 70)")
                    logging.info(f"    Compression ratio: {compression_ratio:.2f} ({(1-compression_ratio)*100:.1f}% reduction)")
                    
                    return compressed_data, True, final_size_mb, compression_ratio
                
                current_image = resized_image
                
                output = io.BytesIO()
                resized_image.save(output,
                                 format='JPEG',
                                 quality=70,  # Use reasonable quality for resized image
                                 optimize=True,
                                 progressive=True)
                
                compressed_data = output.getvalue()
                compressed_size = len(compressed_data)
                
                if compressed_size <= max_size_bytes:
                    final_size_mb = compressed_size / (1024 * 1024)
                    compression_ratio = compressed_size / original_size
                    
                    logging.info(f"  Compression with resizing successful: {filename}")
                    logging.info(f"    Original: {original_size_mb:.2f} MB ({image.width}x{image.height})")
                    logging.info(f"    Compressed: {final_size_mb:.2f} MB ({new_width}x{new_height}, quality: 70)")
                    logging.info(f"    Compression ratio: {compression_ratio:.2f} ({(1-compression_ratio)*100:.1f}% reduction)")
                    
                    return compressed_data, True, final_size_mb, compression_ratio
                
                current_image = resized_image
        
        # Last resort: return best attempt even if still over limit
        if best_data:
            final_size_mb = best_size / (1024 * 1024)
            compression_ratio = best_size / original_size
            
            logging.warning(f"  Could not compress {filename} below {max_size_mb}MB limit")
            logging.warning(f"    Best result: {final_size_mb:.2f} MB (compression: {compression_ratio:.2f})")
            logging.warning(f"    This may fail at Cloudinary upload - consider manual compression")
            
            return best_data, True, final_size_mb, compression_ratio
        
        # If all else fails, return original
        logging.error(f"  Compression failed for {filename}, returning original")
        return image_data, False, original_size_mb, 1.0
        
    except Exception as e:
        logging.error(f"  Error compressing {filename}: {e}")
        return image_data, False, len(image_data) / (1024 * 1024), 1.0

# Retry decorator with exponential backoff for network issues
def retry_with_backoff(max_retries=3, backoff_factor=5, exceptions=(ssl.SSLError, TimeoutError, ConnectionError, OSError, Exception)):
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
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    delay = backoff_factor * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Configure HTTP session with proper connection pooling and timeouts
def create_robust_session():
    """
    Create a requests session with proper retry strategy and connection pooling.
    Optimized for multi-processing to avoid SSL errors.
    """
    session = requests.Session()
    
    # SSL-safe retry strategy with longer backoff
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        raise_on_status=False,
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )
    
    # Conservative connection pooling to avoid SSL conflicts
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=1,
        pool_maxsize=1
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

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
            console_handler.stream = open(1, 'w', encoding='utf-8', closefd=False, buffering=1)
        except (OSError, ValueError):
            pass
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    return str(log_filename)

# Process-safe counter and lock for progress tracking
progress_lock = None
progress_counter = None
error_log = None

class UploadCache:
    """Manages the cache of uploaded files to support resume functionality"""
    
    def __init__(self, source_path: str, folder_name: str | None = None, provider: str = 'gdrive', cache_file_path: str | None = None):
        """Initialize cache for a specific source path and provider"""
        self.source_path = source_path
        self.folder_name = folder_name or 'unknown'
        self.provider = provider
        
        if cache_file_path:
            # Use provided cache file path for batch uploads
            self.cache_file = Path(cache_file_path)
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Create a unique cache file name based on source path and provider
            path_hash = hashlib.md5(f"{provider}_{source_path}".encode()).hexdigest()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Use folder name or source identifier in cache filename
            safe_name = self._sanitize_filename(folder_name or source_path.split('/')[-1] or 'cache')
            self.cache_file = CACHE_DIR / f'{provider}_cache_{safe_name}_{timestamp[:8]}.json'
        
        # Load existing cache if available (search for recent cache files)
        self.cache = self._load_cache()
        
        # Track process for atomic writes
        self.write_lock = multiprocessing.Lock() if hasattr(multiprocessing, 'current_process') else None
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        return re.sub(r'[<>:"/\\|?*]', '_', name)[:50]
    
    def _load_cache(self) -> dict:
        """Load the most recent cache file for this source and provider"""
        try:
            # Look for existing cache files matching our pattern
            safe_name = self._sanitize_filename(self.folder_name or self.source_path.split('/')[-1] or 'cache')
            pattern = str(CACHE_DIR / f'{self.provider}_cache_{safe_name}_*.json')
            cache_files = glob.glob(pattern)
            
            if cache_files:
                # Use the most recent cache file
                latest_cache = max(cache_files, key=os.path.getctime)
                self.cache_file = Path(latest_cache)
                
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    print(f"{SYMBOLS['reload']} Loaded cache: {len(cache_data.get('successful_uploads', {}))} successful, {len(cache_data.get('failed_uploads', {}))} failed")
                    return cache_data
                    
        except Exception as e:
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"{SYMBOLS['warning']} Cache loading issue: {e}")
        
        return {
            'provider': self.provider,
            'source_path': self.source_path,
            'folder_name': self.folder_name,
            'last_run': '',
            'successful_uploads': {},
            'failed_uploads': {}
        }
    
    def _save_cache(self):
        """Save the cache to file with atomic writes"""
        if self.write_lock and not multiprocessing.current_process().name == 'MainProcess':
            # Skip cache writes in worker processes to avoid corruption
            return
            
        try:
            # Use atomic write with temporary file
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            
            # Atomically replace the original file
            temp_file.replace(self.cache_file)
            
        except Exception as e:
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"{SYMBOLS['warning']} Cache save warning: {e}")
    
    def is_uploaded(self, file_id: str) -> bool:
        """Check if a file was successfully uploaded in previous runs"""
        return file_id in self.cache.get('successful_uploads', {})
    
    def mark_uploaded(self, file_id: str, result: dict):
        """Mark a file as successfully uploaded"""
        try:
            self.cache['last_run'] = datetime.now().isoformat()
            self.cache['successful_uploads'][file_id] = {
                'timestamp': datetime.now().isoformat(),
                'cloudinary_url': result.get('cloudinary_url', ''),
                'public_id': result.get('public_id', ''),
                'provider': self.provider
            }
            if file_id in self.cache.get('failed_uploads', {}):
                del self.cache['failed_uploads'][file_id]
            self._save_cache()
        except Exception as e:
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"{SYMBOLS['warning']} Cache update warning: {e}")
    
    def mark_failed(self, file_id: str, error: str):
        """Mark a file as failed upload"""
        try:
            self.cache['last_run'] = datetime.now().isoformat()
            if 'failed_uploads' not in self.cache:
                self.cache['failed_uploads'] = {}
            self.cache['failed_uploads'][file_id] = {
                'timestamp': datetime.now().isoformat(),
                'error': str(error),
                'provider': self.provider
            }
            self._save_cache()
        except Exception as e:
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"{SYMBOLS['warning']} Cache update warning: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'successful': len(self.cache.get('successful_uploads', {})),
            'failed': len(self.cache.get('failed_uploads', {})),
            'last_run': self.cache.get('last_run', 'Never'),
            'provider': self.cache.get('provider', self.provider)
        }

# =============================================================================
# GOOGLE DRIVE INTEGRATION
# =============================================================================

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
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials-bzc.json', GDRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def extract_folder_id_from_url(url_or_id):
    """Extract Google Drive folder ID from various URL formats or return the ID if already provided."""
    if not url_or_id:
        return None
    
    # If it's already just an ID (no slashes or protocols), return it
    if not ('/' in url_or_id or 'http' in url_or_id.lower()):
        return url_or_id
    
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

def get_images_from_gdrive_folder(service, folder_id, recursive=True, parent_path='', folder_name=''):
    """Get list of image files from a Google Drive folder."""
    image_mimetypes = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
        'image/bmp', 'image/webp', 'image/svg+xml'
    }
    image_files = []
    
    try:
        # Get files in current folder
        page_token = None
        while True:
            query = f"'{folder_id}' in parents and trashed=false"
            results = service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, size, parents)",
                pageToken=page_token
            ).execute()
            
            items = results.get('files', [])
            
            for item in items:
                if item['mimeType'] in image_mimetypes:
                    image_files.append({
                        'id': item['id'],
                        'name': item['name'],
                        'mimeType': item['mimeType'],
                        'size': item.get('size', '0'),
                        'folder_path': parent_path
                    })
                elif item['mimeType'] == 'application/vnd.google-apps.folder' and recursive:
                    # Recursively get images from subfolders
                    subfolder_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
                    subfolder_images = get_images_from_gdrive_folder(
                        service, item['id'], recursive=True, 
                        parent_path=subfolder_path, folder_name=item['name']
                    )
                    image_files.extend(subfolder_images)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
                
    except Exception as e:
        print(f"Error accessing Google Drive folder: {e}")
    
    return image_files

@retry_with_backoff(max_retries=3, backoff_factor=2)
def get_google_drive_download_url(service, file_id):
    """Get Google Drive download URL"""
    try:
        file_metadata = service.files().get(fileId=file_id, fields='webContentLink').execute()
        return file_metadata.get('webContentLink')
    except Exception as e:
        raise Exception(f"Failed to get download URL: {str(e)}")

# =============================================================================
# DROPBOX INTEGRATION  
# =============================================================================

def authenticate_dropbox():
    """Authenticate and return Dropbox client"""
    token = os.getenv('DROPBOX_TOKEN')
    if not token:
        raise Exception("DROPBOX_TOKEN not found in environment variables")
    return dropbox.Dropbox(token)

def get_images_from_dropbox_folder(dbx, folder_path):
    """Get list of image files from a Dropbox folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    image_files = []
    
    try:
        result = dbx.files_list_folder(folder_path)
        
        while True:
            for entry in result.entries:
                if hasattr(entry, 'name') and hasattr(entry, 'path_display'):
                    if hasattr(entry, 'size'):  # FileMetadata
                        if os.path.splitext(entry.name)[1].lower() in image_extensions:
                            image_files.append({
                                'id': entry.path_display,  # Use path as ID for Dropbox
                                'name': entry.name,
                                'path': entry.path_display,
                                'size': str(entry.size),
                                'folder_path': ''  # Dropbox uses flat structure for now
                            })
                    else:  # FolderMetadata
                        # Recursively get images from subfolders
                        subfolder_images = get_images_from_dropbox_folder(dbx, entry.path_display)
                        image_files.extend(subfolder_images)
            
            if not result.has_more:
                break
            
            result = dbx.files_list_folder_continue(result.cursor)
        
    except Exception as e:
        print(f"Error accessing Dropbox folder '{folder_path}': {str(e)}")
    
    return image_files

def batch_upload_from_csv(csv_file, provider='gdrive', max_workers=None, recursive=True, force_rescan=False, retry_mode='auto', target_format=None, quality=None):
    """Upload multiple cloud storage folders to Cloudinary from a CSV file.
    
    CSV Format:
        folder_name,link
        Destination Name 1,FOLDER_ID_1
        Destination Name 2,FOLDER_ID_2
    
    Args:
        csv_file (str): Path to CSV file containing folder_name and link columns
        provider (str): 'gdrive' or 'dropbox'
        max_workers (int): Number of worker processes per upload
        recursive (bool): If True, scan subfolders recursively
        force_rescan (bool): If True, ignore cached folder scans
    
    Returns:
        dict: Summary of batch upload results
    """
    print(f"\n{'='*80}")
    print(f"BATCH UPLOAD FROM CSV - {provider.upper()}")
    print(f"{'='*80}\n")
    
    # Check if CSV file exists
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_file}")
        return None
    
    # Read CSV file
    folders_to_upload = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)  # Use regular reader instead of DictReader
            for row_num, row in enumerate(reader, 1):
                # Skip header row if it looks like headers
                if row_num == 1 and len(row) >= 2 and row[1].lower() in ['link', 'url', 'folder_url', 'drive_url']:
                    print(f"📋 Skipping header row: {', '.join(row)}")
                    continue
                    
                if len(row) >= 2:  # Ensure at least 2 columns (EAN, URL)
                    folder_name = row[0].strip()  # Use EAN as folder name
                    folder_url = row[1].strip()   # URL in second column
                    
                    if folder_name and folder_url:
                        if provider == 'gdrive':
                            # Extract Google Drive ID from URL if needed
                            folder_id = extract_folder_id_from_url(folder_url)
                        else:
                            folder_id = folder_url
                        
                        folders_to_upload.append({
                            'folder_name': folder_name,
                            'source_id': folder_id
                        })
                else:
                    print(f"⚠️  Skipping row {row_num}: insufficient columns")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None
    
    if not folders_to_upload:
        print(f"❌ No folders found in CSV file")
        return None
    
    print(f"📊 Found {len(folders_to_upload)} folders to upload:")
    for i, folder in enumerate(folders_to_upload, 1):
        print(f"  {i}. {folder['folder_name']} (ID: {folder['source_id']})")
    print()
    
    # Batch upload summary
    batch_results = {
        'total': len(folders_to_upload),
        'completed': 0,
        'failed': 0,
        'folders': []
    }
    
    # Create consolidated file names for batch upload
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_output_file = f"data/output/batch_{provider}_{batch_timestamp}.csv"
    batch_cache_file = f"data/cache/batch_{provider}_{batch_timestamp}_cache.json"
    batch_log_file = f"data/log/batch_{provider}_{batch_timestamp}.log"
    
    # Initialize consolidated results list
    all_results = []
    
    batch_start_time = time.time()
    
    # Upload each folder sequentially
    for i, folder in enumerate(folders_to_upload, 1):
        print(f"\n{'='*80}")
        print(f"UPLOADING FOLDER {i}/{len(folders_to_upload)}: {folder['folder_name']}")
        print(f"{'='*80}\n")
        
        folder_start_time = time.time()
        
        try:
            results = upload_cloud_folder_to_cloudinary(
                source_path=folder['source_id'],
                provider=provider,
                folder_name=folder['folder_name'],
                max_workers=max_workers,
                recursive=recursive,
                retry_mode=retry_mode,
                output_file=batch_output_file,
                cache_file=batch_cache_file,
                log_file=batch_log_file,
                append_mode=True if i > 1 else False,  # Append mode for subsequent folders
                target_format=target_format,
                quality=quality
            )
            
            folder_time = time.time() - folder_start_time
            
            if results:
                # Add folder name to each result for tracking
                for result in results:
                    result['batch_folder'] = folder['folder_name']
                all_results.extend(results)
                
                successful = sum(1 for r in results if r.get('status') == 'success')
                failed = sum(1 for r in results if r.get('status') == 'failed')
                skipped = sum(1 for r in results if r.get('status') == 'skipped')
                
                batch_results['folders'].append({
                    'name': folder['folder_name'],
                    'status': 'success',
                    'uploaded': successful,
                    'failed': failed,
                    'skipped': skipped,
                    'time': folder_time
                })
                batch_results['completed'] += 1
                
                print(f"\n✅ Folder '{folder['folder_name']}' completed in {folder_time:.2f}s")
                print(f"   📊 {successful} uploaded, {failed} failed, {skipped} skipped")
            else:
                raise Exception("No results returned")
                
        except Exception as e:
            folder_time = time.time() - folder_start_time
            print(f"\n❌ Folder '{folder['folder_name']}' failed: {e}")
            
            batch_results['folders'].append({
                'name': folder['folder_name'],
                'status': 'failed',
                'error': str(e),
                'time': folder_time
            })
            batch_results['failed'] += 1
    
    # Print batch summary
    batch_elapsed = time.time() - batch_start_time
    
    print(f"\n{'='*80}")
    print(f"BATCH UPLOAD SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total folders: {batch_results['total']}")
    print(f"✅ Completed: {batch_results['completed']}")
    print(f"❌ Failed: {batch_results['failed']}")
    print(f"⏱️  Total time: {batch_elapsed:.2f} seconds ({batch_elapsed/60:.1f} minutes)")
    
    if batch_results['completed'] > 0:
        avg_time = sum(f['time'] for f in batch_results['folders'] if f['status'] == 'success') / batch_results['completed']
        print(f"📊 Average time per folder: {avg_time:.2f} seconds")
        
        # Show consolidated files created
        print(f"\n📁 Consolidated files created:")
        print(f"  📊 Results: {batch_output_file}")
        print(f"  🗄️  Cache: {batch_cache_file}")
        print(f"  📋 Logs: {batch_log_file}")
    
    # Show failed folders if any
    if batch_results['failed'] > 0:
        print(f"\n❌ Failed folders:")
        for folder in batch_results['folders']:
            if folder['status'] == 'failed':
                print(f"  - {folder['name']}: {folder.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}\n")
    
    return batch_results

@retry_with_backoff(max_retries=3, backoff_factor=2)
def get_dropbox_download_url(dbx, file_path):
    """Get Dropbox temporary download URL"""
    try:
        tmp_link_resp = dbx.files_get_temporary_link(file_path)
        return tmp_link_resp.link
    except Exception as e:
        raise Exception(f"Failed to get Dropbox download URL: {str(e)}")
        items = results.get('files', [])
        
        for item in items:
            folder_info = {
                'id': item['id'],
                'name': item['name'],
                'path': f"{parent_path}/{item['name']}" if parent_path else item['name'],
                'indent': indent
            }
            folders.append(folder_info)
            
            # Display with indentation
            print(f"{'  ' * indent}📁 {folder_info['path']} (ID: {item['id']})")
            
            # Recursively list subfolders
            subfolders = list_all_google_drive_folders(service, item['id'], indent + 1, folder_info['path'])
            folders.extend(subfolders)
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{parent_id}': {str(e)}")
    
    return folders

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
            print(f"{'  ' * indent}{SYMBOLS['folder']} {folder_path} (ID: {item['id']})")
            
            # Recursively list subfolders
            subfolders = list_all_google_drive_folders(service, item['id'], indent + 1, folder_path)
            folders.extend(subfolders)
        
    except Exception as e:
        print(f"Error accessing Google Drive folder '{parent_id}': {str(e)}")
    
    return folders

def list_shared_drives(service):
    """List all Shared Drives (Team Drives) the user has access to."""
    shared_drives = []
    
    try:
        page_token = None
        
        while True:
            params = {'pageSize': 100}
            if page_token:
                params['pageToken'] = page_token
                
            result = service.drives().list(**params).execute()
            drives = result.get('drives', [])
            
            for drive in drives:
                drive_info = {
                    'id': drive['id'],
                    'name': drive['name']
                }
                shared_drives.append(drive_info)
                print(f"🚗 {drive['name']} (ID: {drive['id']})")
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing shared drives: {str(e)}")
        if "drives" in str(e).lower():
            print("💡 Shared Drives API might not be enabled or you don't have access to any Shared Drives")
    
    return shared_drives

def list_folders_in_shared_drive(service, drive_id, drive_name):
    """List all folders in a specific Shared Drive."""
    folders = []
    
    try:
        page_token = None
        
        while True:
            params = {
                'q': f"mimeType='application/vnd.google-apps.folder' and trashed=false",
                'driveId': drive_id,
                'corpora': 'drive',
                'includeItemsFromAllDrives': True,
                'supportsAllDrives': True,
                'pageSize': 100,
                'fields': "nextPageToken, files(id, name, parents)"
            }
            if page_token:
                params['pageToken'] = page_token
                
            result = service.files().list(**params).execute()
            items = result.get('files', [])
            
            for item in items:
                folder_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'drive_name': drive_name,
                    'drive_id': drive_id
                }
                folders.append(folder_info)
                print(f"  📁 {item['name']} (ID: {item['id']})")
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing folders in shared drive '{drive_name}': {str(e)}")
    
    return folders

def list_shared_with_me(service):
    """List all files and folders shared with me."""
    shared_items = {'folders': [], 'files': []}
    
    try:
        # Query for items shared with me
        query = "sharedWithMe=true and trashed=false"
        page_token = None
        
        while True:
            params = {
                'q': query,
                'pageSize': 100,
                'fields': "nextPageToken, files(id, name, mimeType, owners)"
            }
            if page_token:
                params['pageToken'] = page_token
                
            result = service.files().list(**params).execute()
            items = result.get('files', [])
            
            for item in items:
                owner_name = item.get('owners', [{}])[0].get('displayName', 'Unknown')
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    folder_info = {
                        'id': item['id'],
                        'name': item['name'],
                        'owner': owner_name
                    }
                    shared_items['folders'].append(folder_info)
                    print(f"📁 {item['name']} (ID: {item['id']}) - shared by {owner_name}")
                else:
                    file_info = {
                        'id': item['id'],
                        'name': item['name'],
                        'owner': owner_name
                    }
                    shared_items['files'].append(file_info)
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
    except Exception as e:
        print(f"Error accessing shared files: {str(e)}")
    
    return shared_items

def list_all_shared_content(service):
    """List all shared content including both files shared with me and Shared Drives."""
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
            folder['drive_name'] = drive['name']
        shared_drive_folders.extend(folders)
    
    all_shared['folders'].extend(shared_drive_folders)
    
    return all_shared

# =============================================================================
# CLOUDINARY INTEGRATION
# =============================================================================

def check_cloudinary_folder_exists(folder_name):
    """Check if a folder exists in Cloudinary using the fast folders API."""
    try:
        # Use the fast folders API to check if folder exists
        try:
            result = cloudinary.api.subfolders(folder_name, max_results=1)
            if result.get('folders') or result.get('total_count', 0) > 0:
                # Get a sample of resources to show user what exists
                sample_resources = cloudinary.api.resources(
                    type='upload',
                    prefix=folder_name,
                    max_results=3
                )
                sample_urls = [r['secure_url'] for r in sample_resources.get('resources', [])]
                resource_count = sample_resources.get('total_count', 0)
                return True, resource_count, sample_urls
            return False, 0, []
        except Exception as e:
            if 'does not exist' in str(e) or 'not found' in str(e).lower():
                return False, 0, []
            # Try alternative method - search resources
            try:
                result = cloudinary.api.resources(
                    type='upload',
                    prefix=folder_name,
                    max_results=1
                )
                if result.get('resources'):
                    total_result = cloudinary.api.resources(
                        type='upload',
                        prefix=folder_name,
                        max_results=3
                    )
                    sample_urls = [r['secure_url'] for r in total_result.get('resources', [])]
                    resource_count = total_result.get('total_count', 0)
                    return True, resource_count, sample_urls
                return False, 0, []
            except:
                return False, 0, []
    except Exception as e:
        print(f"Warning: Could not check Cloudinary folder existence: {e}")
        return False, 0, []

def list_cloudinary_folders():
    """List all folders in Cloudinary using the efficient folders API."""
    try:
        print("📊 Scanning Cloudinary for existing folders...")
        all_folders = {}
        
        try:
            # Use the folders API for much faster folder listing
            result = cloudinary.api.root_folders(max_results=500)
            
            for folder in result.get('folders', []):
                folder_path = folder['path']
                all_folders[folder_path] = {
                    'path': folder_path,
                    'name': folder['name'],
                    'has_subfolders': len(folder.get('subfolders', [])) > 0
                }
                
        except Exception as e:
            print(f"Note: Folders API not available: {e}")
            print("Falling back to resource scanning...")
            return list_cloudinary_folders_legacy()
        
        print(f"📁 Found {len(all_folders)} folders in Cloudinary")
        print()
        
        # Display results
        if all_folders:
            print("📋 Your Cloudinary folders:")
            for i, (folder_path, info) in enumerate(sorted(all_folders.items()), 1):
                subfolder_indicator = " (has subfolders)" if info['has_subfolders'] else ""
                print(f"  {i:3d}. {folder_path}{subfolder_indicator}")
                if i >= 50:  # Limit display for very large accounts
                    remaining = len(all_folders) - 50
                    if remaining > 0:
                        print(f"  ... and {remaining} more folders")
                    break
        else:
            print("📭 No folders found in Cloudinary")
            print("💡 Upload some images to create your first folder!")
        
        return all_folders
        
    except Exception as e:
        print(f"❌ Error accessing Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def list_cloudinary_folders_legacy():
    """Legacy method: List all folders by analyzing all resources (slower but comprehensive)."""
    try:
        print("📊 Using legacy method: Scanning all Cloudinary resources...")
        all_folders = {}
        next_cursor = None
        total_resources = 0
        
        while True:
            params = {
                'type': 'upload',
                'max_results': 500,
                'resource_type': 'image'
            }
            if next_cursor:
                params['next_cursor'] = next_cursor
            
            try:
                result = cloudinary.api.resources(**params)
            except Exception as e:
                print(f"Error fetching resources: {e}")
                break
            
            resources = result.get('resources', [])
            if not resources:
                break
            
            total_resources += len(resources)
            
            for resource in resources:
                public_id = resource.get('public_id', '')
                if '/' in public_id:
                    # Extract folder path
                    folder_parts = public_id.split('/')
                    for i in range(1, len(folder_parts)):
                        folder_path = '/'.join(folder_parts[:i])
                        if folder_path and folder_path not in all_folders:
                            all_folders[folder_path] = {
                                'path': folder_path,
                                'name': folder_parts[i-1],
                                'sample_url': resource.get('secure_url'),
                                'resource_count': 0
                            }
                        if folder_path in all_folders:
                            all_folders[folder_path]['resource_count'] += 1
            
            next_cursor = result.get('next_cursor')
            if not next_cursor:
                break
            
            # Progress update for large accounts
            if total_resources % 1000 == 0:
                print(f"  Scanned {total_resources} resources...")
        
        print(f"✅ Legacy scan complete: {total_resources} resources analyzed")
        print(f"📁 Found {len(all_folders)} folders in Cloudinary")
        print()
        
        # Display results
        if all_folders:
            print("📋 Your Cloudinary folders:")
            for i, (folder_path, info) in enumerate(sorted(all_folders.items()), 1):
                count = info.get('resource_count', 0)
                count_text = f" ({count} images)" if count > 0 else ""
                print(f"  {i:3d}. {folder_path}{count_text}")
                if i >= 50:  # Limit display
                    remaining = len(all_folders) - 50
                    if remaining > 0:
                        print(f"  ... and {remaining} more folders")
                    break
        else:
            print("📭 No folders found in Cloudinary")
            print("💡 Upload some images to create your first folder!")
        
        return all_folders
        
    except Exception as e:
        print(f"❌ Error accessing Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def search_cloudinary_folder(search_term):
    """Search for specific folders in Cloudinary by name."""
    try:
        print(f"🔍 Searching Cloudinary for folders matching: '{search_term}'")
        all_folders = {}
        
        try:
            # Use the folders API for much faster folder listing
            result = cloudinary.api.root_folders(max_results=500)
            
            for folder in result.get('folders', []):
                folder_path = folder['path']
                all_folders[folder_path] = {
                    'path': folder_path,
                    'name': folder['name']
                }
                
        except Exception as e:
            print(f"Note: Folders API not available: {e}")
            return {}
        
        # Filter folders that match the search term
        search_term_lower = search_term.lower()
        matching_folders = {}
        
        for folder_path, info in all_folders.items():
            if search_term_lower in folder_path.lower() or search_term_lower in info['name'].lower():
                matching_folders[folder_path] = info
        
        print(f"✅ Search complete: {len(all_folders)} folders scanned")
        print()
        
        # Display results
        if matching_folders:
            print(f"🎯 Found {len(matching_folders)} folders matching '{search_term}':")
            for i, (folder_path, info) in enumerate(sorted(matching_folders.items()), 1):
                print(f"  {i:3d}. {folder_path}")
        else:
            print(f"📭 No folders found matching '{search_term}'")
            print("💡 Try a different search term or check the exact folder name")
        
        return matching_folders
        
    except Exception as e:
        print(f"❌ Error searching Cloudinary: {e}")
        print("💡 Please check your Cloudinary configuration in .env file")
        return {}

def prompt_folder_action(folder_name, resource_count, sample_urls):
    """Prompt user for action when folder already exists in Cloudinary."""
    print(f"\n⚠️  Folder '{folder_name}' already exists in Cloudinary!")
    print(f"   📊 Contains {resource_count} existing images")
    
    if sample_urls:
        print(f"   🖼️  Sample images:")
        for i, url in enumerate(sample_urls[:3], 1):
            print(f"      {i}. {url}")
        if resource_count > 3:
            print(f"      ... and {resource_count - 3} more")
    
    print(f"\n🤔 What would you like to do?")
    print(f"   1. MERGE - Add new images to existing folder (skip duplicates)")
    print(f"   2. RENAME - Use a different folder name")
    print(f"   3. CANCEL - Stop the upload")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            return 'merge', folder_name
        elif choice == '2':
            while True:
                new_name = input("Enter new folder name: ").strip()
                if new_name and new_name != folder_name:
                    # Check if the new name also exists
                    exists, count, urls = check_cloudinary_folder_exists(new_name)
                    if exists:
                        print(f"⚠️  Folder '{new_name}' also exists with {count} images. Try another name.")
                        continue
                    return 'rename', new_name
                elif new_name == folder_name:
                    print("Please choose a different name.")
                else:
                    print("Please enter a valid folder name.")
        elif choice == '3':
            return 'cancel', None
        else:
            print("Please enter 1, 2, or 3")

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
            return False, "Failed to authenticate Google Drive"
        
        # Test the connection by listing the root folder
        results = service.files().list(pageSize=1).execute()
        return True, "Google Drive connection successful"
    except Exception as e:
        return False, f"Google Drive connection error: {str(e)}"

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

def sanitize_cloudinary_public_id(text):
    """Sanitize text for use in Cloudinary public_id."""
    if not text:
        return "untitled"
    
    # First, handle accents by converting to ASCII equivalents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Replace problematic characters
    text = text.replace('&', 'and')
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-zA-Z0-9\-_/]', '', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_/')
    
    if not text:
        return "untitled"
    
    return text

# =============================================================================
# UNIFIED UPLOAD WORKER
# =============================================================================

def _upload_worker(file_info, base_folder_name, source_id, folder_name, provider, p_lock, p_counter, p_error_log, target_format=None, quality=None):
    """
    Worker function for multiprocessing.
    Handles both Google Drive and Dropbox uploads.
    """
    # Set global variables for this process
    global progress_lock, progress_counter, error_log
    progress_lock = p_lock
    progress_counter = p_counter
    error_log = p_error_log
    
    try:
        # Initialize provider service for this process
        if provider == 'gdrive':
            service = authenticate_google_drive()
            if not service:
                raise Exception("Failed to authenticate Google Drive")
        elif provider == 'dropbox':
            service = authenticate_dropbox()
        else:
            raise Exception(f"Unknown provider: {provider}")
        
        # Each process needs its own cache instance
        cache = UploadCache(source_id, folder_name, provider)
        
        return upload_single_image_from_cloud(service, file_info, base_folder_name, cache, provider, target_format, quality)
        
    except Exception as e:
        # Handle worker initialization failures
        error_message = f"Worker initialization failed: {str(e)}"
        with progress_lock:
            progress_counter['failed'] += 1
            if len(error_log) < 10:
                error_log.append(f"{file_info.get('name', 'unknown')}: {error_message}")
        
        return {
            'local_filename': file_info.get('name', 'unknown'),
            'cloudinary_url': 'UPLOAD_FAILED',
            'jpg_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message,
            'provider': provider,
            'was_compressed': False,
            'final_size_mb': None,
            'compression_ratio': None
        }

def upload_single_image_from_cloud(service, file_info, base_folder_name, cache, provider, target_format=None, quality=None):
    """
    Cloud-to-cloud upload: Get a temporary download URL and let Cloudinary fetch it directly.
    Enhanced for multiprocessing - each process creates its own session.
    """
    file_id = file_info['id']
    filename = file_info['name']
    folder_path = file_info.get('folder_path', '')
    
    # Create a session for this process
    session = create_robust_session()
    
    # Create the full Cloudinary folder path
    cloudinary_folder = base_folder_name.strip().rstrip('/')
    cloudinary_folder = re.sub(r'\s*/\s*', '/', cloudinary_folder)
    cloudinary_folder = re.sub(r'\s+', ' ', cloudinary_folder)
    cloudinary_folder = cloudinary_folder.strip()
    
    # Already uploaded? return cached result and update progress
    if cache.is_uploaded(file_id):
        if progress_lock and progress_counter:
            with progress_lock:
                progress_counter['skipped'] += 1
                current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']
        
        cached_data = cache.cache['successful_uploads'][file_id]
        return {
            'local_filename': os.path.splitext(filename)[0],
            'cloudinary_url': cached_data.get('cloudinary_url', ''),
            'status': 'skipped',
            'public_id': cached_data.get('public_id', ''),
            'provider': provider
        }

    try:
        # Get download URL based on provider
        if provider == 'gdrive':
            download_url = get_google_drive_download_url(service, file_id)
        elif provider == 'dropbox':
            download_url = get_dropbox_download_url(service, file_id)
        else:
            raise Exception(f"Unknown provider: {provider}")
        
        if not download_url:
            raise Exception("Failed to get download URL")
        
        # Ensure download_url is a string
        if isinstance(download_url, list):
            download_url = download_url[0] if download_url else None
        
        if not download_url or not isinstance(download_url, str):
            raise Exception("Invalid download URL received")
        
        # Download the image data
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        original_data = response.content
        
        # Check and compress if necessary
        compressed_data, was_compressed, final_size_mb, compression_ratio = compress_image_for_cloudinary(
            original_data, filename
        )
        
        # Prepare Cloudinary upload
        file_stem, ext = os.path.splitext(filename)
        original_extension = ext.lower().replace('.', '') if ext else 'jpg'
        public_id = sanitize_cloudinary_public_id(file_stem)
        
        # Upload compressed data to Cloudinary
        response = cloudinary.uploader.upload(
            compressed_data,
            folder=cloudinary_folder,
            public_id=public_id,
            use_filename=USE_FILENAME,
            unique_filename=UNIQUE_FILENAME,
            overwrite=True,
            format=original_extension,
            resource_type="image"
        )
        
        # Generate JPG URL for the uploaded image
        cloudinary_url = response['secure_url']
        
        # Apply format conversion if requested
        if target_format:
            converted_url = convert_cloudinary_url_with_quality(cloudinary_url, target_format, quality)
        else:
            converted_url = convert_cloudinary_url_to_jpg(cloudinary_url)
        
        result = {
            'local_filename': file_stem,
            'cloudinary_url': cloudinary_url,
            'jpg_url': converted_url,
            'status': 'success',
            'public_id': response.get('public_id', ''),
            'provider': provider,
            'was_compressed': was_compressed,
            'final_size_mb': round(final_size_mb, 2),
            'compression_ratio': round(compression_ratio, 2) if was_compressed else None,
            'target_format': target_format,
            'quality': quality
        }

        # Cache and update progress
        cache.mark_uploaded(file_id, result)
        if progress_lock and progress_counter:
            with progress_lock:
                progress_counter['uploaded'] += 1
                current = progress_counter['uploaded'] + progress_counter['failed'] + progress_counter['skipped']

        return result

    except Exception as e:
        error_message = str(e)
        if progress_lock and progress_counter and error_log is not None:
            with progress_lock:
                progress_counter['failed'] += 1
                if len(error_log) < 10:
                    error_log.append(f"{filename}: {error_message}")

        file_stem = os.path.splitext(filename)[0]
        cache.mark_failed(file_id, error_message)
        return {
            'local_filename': file_stem,
            'cloudinary_url': 'UPLOAD_FAILED',
            'jpg_url': 'UPLOAD_FAILED', 
            'status': 'failed',
            'error': error_message,
            'provider': provider,
            'was_compressed': False,
            'final_size_mb': None,
            'compression_ratio': None
        }

# =============================================================================
# MAIN UPLOAD FUNCTION
# =============================================================================

def upload_cloud_folder_to_cloudinary(source_path, provider, folder_name=None, max_workers=None, recursive=True, retry_mode='auto', output_file=None, cache_file=None, log_file=None, append_mode=False, force_rescan=False, target_format=None, quality=None):
    """
    Upload images from a cloud storage folder to Cloudinary using multiprocessing.
    
    Args:
        source_path (str): Google Drive folder ID/URL or Dropbox folder path
        provider (str): 'gdrive' or 'dropbox'
        folder_name (str): Optional custom folder name for Cloudinary
        max_workers (int): Number of concurrent worker processes
        recursive (bool): If True, scan and upload from subfolders recursively
        retry_mode (str): Retry mode for failed uploads ('auto', 'true', 'false')
    """
    # Test Cloudinary connection
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print("\n⚠️  Please check your config.py and .env file")
        return
    
    print("  ✓ Cloudinary verified\n")
    
    # Initialize provider service
    print(f"Testing {provider.title()} connection...")
    try:
        if provider == 'gdrive':
            service = authenticate_google_drive()
            if not service:
                raise Exception("Failed to authenticate Google Drive")
            
            # Extract folder ID if URL provided
            source_id = extract_folder_id_from_url(source_path)
            if not source_id:
                raise Exception("Invalid Google Drive folder ID/URL")
            
            # Get folder name if not provided
            if not folder_name:
                try:
                    folder_metadata = service.files().get(fileId=source_id, fields='name').execute()
                    folder_name = folder_metadata.get('name', source_id)
                except:
                    folder_name = source_id
            
        elif provider == 'dropbox':
            service = authenticate_dropbox()
            source_id = source_path  # For Dropbox, use path directly
            
            # Get folder name if not provided
            if not folder_name:
                folder_name = os.path.basename(source_path.rstrip('/'))
            
        else:
            raise Exception(f"Unknown provider: {provider}")
            
        print(f"  ✓ {provider.title()} verified\n")
        
    except Exception as e:
        print(f"  ❌ {provider.title()} connection failed: {e}")
        return
    
    # Get images from source
    print(f"Scanning {provider.title()} folder: {source_path}")
    
    image_files = []
    if provider == 'gdrive':
        image_files = get_images_from_gdrive_folder(service, source_id, recursive=recursive)
    elif provider == 'dropbox':
        image_files = get_images_from_dropbox_folder(service, source_path)
    
    if not image_files:
        print(f"No images found in '{source_path}'")
        return
    
    # Count files by folder for summary
    folder_counts = {}
    for img in image_files:
        folder_path = img.get('folder_path', 'root')
        folder_counts[folder_path] = folder_counts.get(folder_path, 0) + 1
    
    print(f"\n📊 Found {len(image_files)} images across {len(folder_counts)} folder(s):")
    for folder_path, count in sorted(folder_counts.items()):
        print(f"  {SYMBOLS['folder']} {folder_path}: {count} images")
    print()
    
    # Setup logging
    if log_file:
        # Use provided log file for batch upload
        log_filename = log_file
    else:
        # Use individual log file for single folder upload
        log_filename = setup_logging(folder_name)
    
    # Generate CSV filename
    if output_file:
        # Use provided output file for batch upload
        output_csv = Path(output_file)
        # Ensure output directory exists
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use individual output file for single folder upload
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_csv = OUTPUT_DIR / f"{provider}_{safe_folder_name}_{timestamp}.csv"
    
    # Initialize upload cache
    if cache_file:
        # Use provided cache file for batch upload
        cache = UploadCache(source_id, folder_name, provider, cache_file_path=cache_file)
    else:
        # Use individual cache file for single folder upload
        cache = UploadCache(source_id, folder_name, provider)
    cache_stats = cache.get_stats()
    
    if cache_stats['successful'] > 0:
        print(f"{SYMBOLS['reload']} Cache loaded: {cache_stats['successful']} files already uploaded")
    else:
        print(f"{SYMBOLS['new']} Starting fresh upload")
    
    # Initialize multiprocessing Manager for shared state
    manager = Manager()
    global progress_lock, progress_counter, error_log
    progress_lock = manager.Lock()
    progress_counter = manager.dict()
    error_log = manager.list()
    
    # Initialize progress counter
    progress_counter['total'] = len(image_files)
    progress_counter['uploaded'] = 0
    progress_counter['failed'] = 0
    progress_counter['skipped'] = 0
    
    # Determine number of worker processes
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Using {max_workers} worker processes for parallel uploads")
    
    logging.info(f"Processing {provider} folder: {folder_name}")
    logging.info(f"Source: {source_path}")
    logging.info(f"Recursive scanning: {recursive}")
    logging.info(f"Found {len(image_files)} images to process")
    logging.info(f"Using multiprocessing with {max_workers} worker processes")
    
    start_time = time.time()
    results = []
    
    # Multiprocessing for maximum performance
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_image = {
            executor.submit(
                _upload_worker, 
                img, 
                folder_name, 
                source_id, 
                folder_name,
                provider,
                progress_lock, 
                progress_counter, 
                error_log,
                target_format,
                quality
            ): img 
            for img in image_files
        }
        
        # Collect results with progress bar
        with tqdm(total=len(image_files), desc="Uploading images") as pbar:
            for future in as_completed(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    img = future_to_image[future]
                    print(f"❌ Unexpected error processing {img.get('name', 'unknown')}: {e}")
                    results.append({
                        'local_filename': img.get('name', 'unknown'),
                        'cloudinary_url': 'UPLOAD_FAILED',
                        'jpg_url': 'UPLOAD_FAILED',
                        'status': 'failed',
                        'error': str(e),
                        'provider': provider,
                        'was_compressed': False,
                        'final_size_mb': None,
                        'compression_ratio': None
                    })
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Handle retry logic for failed uploads within executor context
    failed_results = [r for r in results if r['status'] == 'failed']
    should_retry = False
    
    if failed_results and retry_mode in ['auto', 'true']:
        if retry_mode == 'auto':
            # Auto mode: retry if there are failed uploads but not too many
            should_retry = len(failed_results) < len(image_files) * 0.5  # Retry if <50% failed
        elif retry_mode == 'true':
            should_retry = True
            
        if should_retry:
            print(f"\n🔄 RETRY MODE: {retry_mode} - Found {len(failed_results)} failed uploads")
            print(f"🔄 Retrying failed uploads...")
            
            # Extract failed image info for retry
            retry_files = []
            for result in failed_results:
                # Find original image info
                for img in image_files:
                    if (os.path.splitext(img.get('name', ''))[0] == result['local_filename'] or 
                        img.get('name', '') == result['local_filename']):
                        retry_files.append(img)
                        break
            
            if retry_files:
                print(f"🔄 Retrying {len(retry_files)} failed uploads...")
                
                # Retry upload with same executor
                with ProcessPoolExecutor(max_workers=max_workers) as retry_executor:
                    retry_future_to_image = {
                        retry_executor.submit(
                            _upload_worker, 
                            img, 
                            folder_name, 
                            source_id, 
                            folder_name,
                            provider,
                            progress_lock, 
                            progress_counter, 
                            error_log
                        ): img 
                        for img in retry_files
                    }
                    
                    # Collect retry results
                    retry_results = []
                    with tqdm(total=len(retry_files), desc="Retrying failed uploads") as retry_pbar:
                        for future in as_completed(retry_future_to_image):
                            try:
                                result = future.result()
                                retry_results.append(result)
                                retry_pbar.update(1)
                            except Exception as e:
                                img = retry_future_to_image[future]
                                retry_results.append({
                                    'local_filename': img.get('name', 'unknown'),
                                    'cloudinary_url': 'RETRY_FAILED',
                                    'jpg_url': 'RETRY_FAILED',
                                    'status': 'retry_failed',
                                    'error': str(e),
                                    'provider': provider,
                                    'was_compressed': False,
                                    'final_size_mb': None,
                                    'compression_ratio': None
                                })
                                retry_pbar.update(1)
                    
                    # Update original results with retry results
                    retry_success = 0
                    for retry_result in retry_results:
                        if retry_result['status'] == 'success':
                            retry_success += 1
                            # Update the original failed result
                            for i, orig_result in enumerate(results):
                                if orig_result['local_filename'] == retry_result['local_filename']:
                                    results[i] = retry_result
                                    break
                    
                    print(f"🔄 Retry completed: {retry_success}/{len(retry_files)} succeeded")
                
                # Retry upload with same executor
                with ProcessPoolExecutor(max_workers=max_workers) as retry_executor:
                    retry_future_to_image = {
                        retry_executor.submit(
                            _upload_worker, 
                            img, 
                            folder_name, 
                            source_id, 
                            folder_name,
                            provider,
                            progress_lock, 
                            progress_counter, 
                            error_log
                        ): img 
                        for img in retry_files
                    }
                    
                    # Collect retry results
                    retry_results = []
                    with tqdm(total=len(retry_files), desc="Retrying failed uploads") as retry_pbar:
                        for future in as_completed(retry_future_to_image):
                            try:
                                result = future.result()
                                retry_results.append(result)
                                retry_pbar.update(1)
                            except Exception as e:
                                img = retry_future_to_image[future]
                                retry_results.append({
                                    'local_filename': img.get('name', 'unknown'),
                                    'cloudinary_url': 'RETRY_FAILED',
                                    'jpg_url': 'RETRY_FAILED',
                                    'status': 'retry_failed',
                                    'error': str(e),
                                    'provider': provider,
                                    'was_compressed': False,
                                    'final_size_mb': None,
                                    'compression_ratio': None
                                })
                                retry_pbar.update(1)
                    
                    # Update original results with retry results
                    retry_success = 0
                    for retry_result in retry_results:
                        if retry_result['status'] == 'success':
                            retry_success += 1
                            # Update the original failed result
                            for i, orig_result in enumerate(results):
                                if orig_result['local_filename'] == retry_result['local_filename']:
                                    results[i] = retry_result
                                    break
                    
                    print(f"🔄 Retry completed: {retry_success}/{len(retry_files)} succeeded")
        elif retry_mode == 'auto':
            print(f"\n⏭️  RETRY MODE: auto - Too many failures ({len(failed_results)}/{len(image_files)}), skipping retry")
    elif failed_results and retry_mode == 'false':
        print(f"\n⏭️  RETRY MODE: false - Skipping retry of {len(failed_results)} failed uploads")
    
    elapsed_time = time.time() - start_time
    
    # Write results to CSV
    if results:
        csv_columns = ['local_filename', 'cloudinary_url', 'jpg_url', 'status', 'provider', 'was_compressed', 'final_size_mb', 'compression_ratio', 'batch_folder', 'target_format', 'quality']
        
        try:
            write_mode = 'a' if append_mode and output_csv.exists() else 'w'
            file_exists = output_csv.exists() and write_mode == 'a'
            
            with open(output_csv, write_mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                
                # Write header only if file is new or being overwritten
                if not file_exists:
                    writer.writeheader()
                
                # Add batch_folder info to results if not present
                for result in results:
                    if 'batch_folder' not in result:
                        result['batch_folder'] = folder_name or ''
                
                writer.writerows(results)
            
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] in ['failed', 'retry_failed'])
            skipped = sum(1 for r in results if r['status'] == 'skipped')
            compressed = sum(1 for r in results if r.get('was_compressed') == True)
            
            print(f"\n{'='*60}")
            print(f"✓ Upload completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                print(f"✓ Average speed: {successful/elapsed_time:.2f} images/second")
            print(f"✓ Results saved to '{output_csv}'")
            print(f"  Total images processed: {len(image_files)}")
            print(f"  Successfully uploaded: {successful}")
            print(f"  Previously uploaded (skipped): {skipped}")
            print(f"  Failed uploads: {failed}")
            
            if compressed > 0:
                print(f"\n{SYMBOLS['compress']} Compression Statistics:")
                print(f"  Images compressed: {compressed}/{successful}")
                avg_compression = sum(r.get('compression_ratio', 1.0) for r in results if r.get('was_compressed')) / max(compressed, 1)
                avg_size = sum(r.get('final_size_mb', 0) for r in results if r.get('final_size_mb')) / max(successful, 1)
                print(f"  Average compression ratio: {avg_compression:.2f}")
                print(f"  Average final size: {avg_size:.2f} MB")
            
            if error_log:
                print(f"\n⚠️  Sample errors (first 10):")
                for err in list(error_log)[:10]:
                    print(f"  - {err}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error writing CSV: {str(e)}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(
        description="Upload images from Google Drive or Dropbox to Cloudinary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  list <provider>           : List folders in cloud storage
  shared                   : List shared Google Drive content (files shared with you)
  drives                   : List Google Drive Shared Drives (Team Drives)
  cloudinary [--folder]    : List/search Cloudinary folders
  upload <provider> <path> : Upload from cloud storage folder
  batch-upload <csv>       : Upload multiple folders from CSV

Options:
  --retry <mode>           : Retry mode for failed uploads (auto/true/false, default: auto)
  --workers <count>        : Number of worker processes (default: CPU count)
  --no-recursive           : Disable recursive folder scanning
  --force-rescan           : Force rescan of folders (ignore cache)

Examples:
  # List folders
  python cloudstorage_tocloudinary.py list gdrive
  python cloudstorage_tocloudinary.py list dropbox
  python cloudstorage_tocloudinary.py shared
  python cloudstorage_tocloudinary.py drives
  
  # Cloudinary management
  python cloudstorage_tocloudinary.py cloudinary
  python cloudstorage_tocloudinary.py cloudinary --folder beauty
  
  # Google Drive uploads
  python cloudstorage_tocloudinary.py upload gdrive "1ABC123def456" --folder "My Photos"
  python cloudstorage_tocloudinary.py upload gdrive "https://drive.google.com/drive/folders/1ABC123def456"
  python cloudstorage_tocloudinary.py upload gdrive "1ABC123" --format webp --quality 80
  
  # Dropbox uploads  
  python cloudstorage_tocloudinary.py upload dropbox "/my_images" --folder "Photos"
  python cloudstorage_tocloudinary.py upload dropbox "/vacation" --workers 8 --format jpg --quality 85
  
  # Batch uploads
  python cloudstorage_tocloudinary.py batch-upload folders.csv --provider gdrive
  python cloudstorage_tocloudinary.py batch-upload dropbox_folders.csv --provider dropbox --format webp
  
  # With retry options
  python cloudstorage_tocloudinary.py upload gdrive "1ABC123" --retry true --format png
  python cloudstorage_tocloudinary.py batch-upload folders.csv --retry false --format jpg --quality 90
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List folders in cloud storage')
    list_parser.add_argument('provider', choices=['gdrive', 'dropbox'], help='Cloud storage provider')
    
    # Shared content commands (Google Drive only)
    shared_parser = subparsers.add_parser('shared', help='List files/folders shared with you (Google Drive)')
    drives_parser = subparsers.add_parser('drives', help='List Google Drive Shared Drives (Team Drives)')
    
    # Cloudinary management
    cloudinary_parser = subparsers.add_parser('cloudinary', help='List/search Cloudinary folders')
    cloudinary_parser.add_argument('--folder', help='Search for specific folder name')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload images from cloud storage')
    upload_parser.add_argument('provider', choices=['gdrive', 'dropbox'], help='Cloud storage provider')
    upload_parser.add_argument('source_path', help='Google Drive folder ID/URL or Dropbox folder path')
    upload_parser.add_argument('--folder', help='Custom Cloudinary folder name (default: use source folder name)')
    upload_parser.add_argument('--workers', type=int, help='Number of worker processes (default: CPU count)')
    upload_parser.add_argument('--no-recursive', action='store_true', help='Don\'t scan subfolders recursively')
    upload_parser.add_argument('--force-rescan', action='store_true', help='Force rescan of folder structure (ignore cache)')
    upload_parser.add_argument('--retry', choices=['auto', 'true', 'false'], default='auto',
                            help='Retry mode for failed uploads (auto/true/false, default: auto)')
    upload_parser.add_argument('--format', choices=['jpg', 'jpeg', 'png', 'webp', 'avif'], 
                            help='Convert images to specified format (jpg, png, webp, etc.)')
    upload_parser.add_argument('--quality', type=int, 
                            help='Quality setting for lossy formats like JPG/WebP (1-100)')
    
    # Batch upload command
    batch_parser = subparsers.add_parser('batch-upload', help='Upload multiple folders from CSV file')
    batch_parser.add_argument('csv_file', help='Path to CSV file with folder_name,link columns')
    batch_parser.add_argument('--provider', choices=['gdrive', 'dropbox'], default='gdrive', help='Cloud storage provider')
    batch_parser.add_argument('--workers', type=int, help='Number of worker processes per upload')
    batch_parser.add_argument('--no-recursive', action='store_true', help='Don\'t scan subfolders recursively')
    batch_parser.add_argument('--force-rescan', action='store_true', help='Force rescan of folder structures (ignore cache)')
    batch_parser.add_argument('--retry', choices=['auto', 'true', 'false'], default='auto',
                            help='Retry mode for failed uploads (auto/true/false, default: auto)')
    batch_parser.add_argument('--format', choices=['jpg', 'jpeg', 'png', 'webp', 'avif'], 
                            help='Convert images to specified format (jpg, png, webp, etc.)')
    batch_parser.add_argument('--quality', type=int, 
                            help='Quality setting for lossy formats like JPG/WebP (1-100)')
    
    # Backwards compatibility: direct provider commands
    gdrive_parser = subparsers.add_parser('gdrive', help='Google Drive upload (shorthand)')
    gdrive_parser.add_argument('source_path', help='Google Drive folder ID or URL')
    gdrive_parser.add_argument('folder_name', nargs='?', help='Custom Cloudinary folder name')
    gdrive_parser.add_argument('workers', nargs='?', type=int, help='Number of worker processes')
    gdrive_parser.add_argument('--no-recursive', action='store_true', help='Don\'t scan subfolders')
    gdrive_parser.add_argument('--force-rescan', action='store_true', help='Force rescan (ignore cache)')
    gdrive_parser.add_argument('--retry', choices=['auto', 'true', 'false'], default='auto', help='Retry mode')
    gdrive_parser.add_argument('--format', choices=['jpg', 'jpeg', 'png', 'webp', 'avif'], 
                            help='Convert images to specified format')
    gdrive_parser.add_argument('--quality', type=int, help='Quality setting (1-100)')
    
    dropbox_parser = subparsers.add_parser('dropbox', help='Dropbox upload (shorthand)')  
    dropbox_parser.add_argument('source_path', help='Dropbox folder path')
    dropbox_parser.add_argument('folder_name', nargs='?', help='Custom Cloudinary folder name')
    dropbox_parser.add_argument('workers', nargs='?', type=int, help='Number of worker processes')
    dropbox_parser.add_argument('--no-recursive', action='store_true', help='Don\'t scan subfolders')
    dropbox_parser.add_argument('--force-rescan', action='store_true', help='Force rescan (ignore cache)')
    dropbox_parser.add_argument('--retry', choices=['auto', 'true', 'false'], default='auto', help='Retry mode')
    dropbox_parser.add_argument('--format', choices=['jpg', 'jpeg', 'png', 'webp', 'avif'], 
                            help='Convert images to specified format')
    dropbox_parser.add_argument('--quality', type=int, help='Quality setting (1-100)')
    
    args = parser.parse_args()
    
    # Validate quality parameter if provided
    if hasattr(args, 'quality') and args.quality:
        if args.quality < 1 or args.quality > 100:
            print("❌ Quality must be between 1 and 100")
            sys.exit(1)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle list command
    if args.command == 'list':
        print(f"Testing {args.provider.title()} connection...")
        
        try:
            if args.provider == 'gdrive':
                service = authenticate_google_drive()
                print("📁 Scanning YOUR own folders in Google Drive...\n")
                folders = list_all_google_drive_folders(service)
                print(f"\n{'='*60}")
                print(f"📁 Your own folders found: {len(folders)}")
                print(f"{'='*60}")
                
            elif args.provider == 'dropbox':
                dbx = authenticate_dropbox()
                print("📁 Scanning all folders in your Dropbox...\n")
                # Add Dropbox folder listing logic here if needed
                print("Dropbox folder listing not yet implemented in combined script")
                
        except Exception as e:
            print(f"Error listing {args.provider} folders: {e}")
            sys.exit(1)
    
    # Handle Google Drive shared content
    elif args.command == 'shared':
        print("Testing Google Drive connection...")
        try:
            service = authenticate_google_drive()
            print("📋 Scanning content shared with you...\n")
            shared_content = list_all_shared_content(service)
            
            print(f"\n{'='*60}")
            print(f"📁 Shared folders found: {len(shared_content['folders'])}")
            print(f"📄 Shared files found: {len(shared_content['files'])}")
            print(f"🚗 Shared Drives found: {len(shared_content['shared_drives'])}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error listing shared content: {e}")
            sys.exit(1)
    
    # Handle Google Drive Shared Drives
    elif args.command == 'drives':
        print("Testing Google Drive connection...")
        try:
            service = authenticate_google_drive()
            print("🚗 Scanning Shared Drives (Team Drives)...\n")
            drives = list_shared_drives(service)
            
            print(f"\n{'='*60}")
            print(f"🚗 Shared Drives found: {len(drives)}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error listing Shared Drives: {e}")
            sys.exit(1)
    
    # Handle Cloudinary management
    elif args.command == 'cloudinary':
        if args.folder:
            search_cloudinary_folder(args.folder)
        else:
            list_cloudinary_folders()
    
    # Handle batch upload
    elif args.command == 'batch-upload':
        batch_upload_from_csv(
            csv_file=args.csv_file,
            provider=args.provider,
            max_workers=args.workers,
            recursive=not getattr(args, 'no_recursive', False),
            force_rescan=getattr(args, 'force_rescan', False),
            retry_mode=getattr(args, 'retry', 'auto'),
            target_format=getattr(args, 'format', None),
            quality=getattr(args, 'quality', None)
        )
    
    # Handle upload commands
    elif args.command in ['upload', 'gdrive', 'dropbox']:
        # Determine provider and source
        if args.command == 'upload':
            provider = args.provider
            source_path = args.source_path
            folder_name = getattr(args, 'folder', None)
            workers = getattr(args, 'workers', None)
        else:
            provider = args.command
            source_path = args.source_path
            folder_name = getattr(args, 'folder_name', None)
            workers = getattr(args, 'workers', None)
        
        # Upload with specified parameters
        upload_cloud_folder_to_cloudinary(
            source_path=source_path,
            provider=provider, 
            folder_name=folder_name,
            max_workers=workers,
            recursive=not getattr(args, 'no_recursive', False),
            retry_mode=getattr(args, 'retry', 'auto'),
            force_rescan=getattr(args, 'force_rescan', False),
            target_format=getattr(args, 'format', None),
            quality=getattr(args, 'quality', None)
        )
    
    else:
        parser.print_help()
        sys.exit(1)