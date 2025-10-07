"""
Dropbox to Cloudinary Uploader
=============================
Downloads images from Dropbox shared folders and uploads them to Cloudinary.
Supports multiple brand folders and maintains organization structure.

Usage: python scripts/dropbox_to_cloudinary.py
"""

import os
import re
import time
import csv
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Tuple, Optional

try:
    import dropbox
except ImportError:
    print("ERROR: dropbox package not installed")
    print("Install with: pip install dropbox")
    exit(1)

try:
    import cloudinary
    import cloudinary.uploader
except ImportError:
    print("ERROR: cloudinary package not installed")
    print("Install with: pip install cloudinary")
    exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional: base Cloudinary folder prefix for organizing uploads
BASE_CLOUDINARY_FOLDER = os.getenv('CLOUDINARY_FOLDER', 'products')

# ----------------------------
# 2) Init SDKs
# ----------------------------
def init_cloudinary():
    """Initialize Cloudinary with credentials from environment"""
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    api_key = os.getenv('CLOUDINARY_API_KEY')
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
    
    if not all([cloud_name, api_key, api_secret]):
        raise ValueError("Missing Cloudinary credentials in .env file")
    
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )

def init_dropbox():
    """Initialize Dropbox with token from environment"""
    token = os.getenv('DROPBOX_TOKEN')
    if not token:
        raise ValueError("Missing DROPBOX_TOKEN in .env file")
    return dropbox.Dropbox(token)

# ----------------------------
# Helpers
# ----------------------------
def normalize_folder_url(shared_url: str) -> str:
    """
    Force dl=0 on folder link (preview page); we'll use it as a 'shared_link' object
    to list contents via Dropbox API. The other params (rlkey, st) are fine to keep.
    """
    parts = urlparse(shared_url)
    qs = parse_qs(parts.query)
    qs["dl"] = ["0"]
    # rebuild query string
    new_query = "&".join([f"{k}={v[0]}" for k, v in qs.items()])
    return f"{parts.scheme}://{parts.netloc}{parts.path}?{new_query}"

def list_files_in_shared_folder(dbx: dropbox.Dropbox, shared_folder_url: str):
    """
    Use Dropbox API to enumerate files inside a shared folder URL.
    Returns a list of (display_path, path_lower, name, id, is_image_or_video).
    """
    from dropbox.files import ListFolderArg, SharedLink, FolderMetadata, FileMetadata
    results = []

    shared_link = SharedLink(url=normalize_folder_url(shared_folder_url))

    # Start at root of the shared link
    cursor = None

    while True:
        if cursor:
            resp = dbx.files_list_folder_continue(cursor)
        else:
            resp = dbx.files_list_folder(ListFolderArg(
                path="", recursive=True, shared_link=shared_link
            ))
        for entry in resp.entries:
            if isinstance(entry, FileMetadata):
                name = entry.name
                # Basic filter: images/videos common extensions
                is_media = bool(re.search(r"\.(jpg|jpeg|png|webp|gif|tif|tiff|bmp|heic|mp4|mov|webm|avi|mkv)$", name, re.I))
                results.append((
                    entry.path_display, getattr(entry, "path_lower", None),
                    entry.name, entry.id, is_media
                ))
        if resp.has_more:
            cursor = resp.cursor
        else:
            break
    return results

def get_direct_file_url(dbx: dropbox.Dropbox, entry_path_lower: str):
    """
    Ask Dropbox for a temporary direct link to the file content.
    These links are valid for ~4 hours and perfect for piping into Cloudinary upload.
    """
    tmp = dbx.files_get_temporary_link(entry_path_lower)
    return tmp.link

def upload_to_cloudinary(url: str, folder_name: str, filename: str):
    """
    Upload the Dropbox direct URL to Cloudinary.
    Keeps original filename (without extension) as public_id where possible.
    """
    # build Cloudinary folder path: e.g., products/folder_name
    folder = f"{BASE_CLOUDINARY_FOLDER}/{folder_name}" if BASE_CLOUDINARY_FOLDER else folder_name

    # Public ID suggestion (strip extension, keep safe chars)
    public_id = re.sub(r"\.[^.]+$", "", filename)
    public_id = re.sub(r"[^a-zA-Z0-9/_-]+", "-", public_id).strip("-")

    params = {
        "folder": folder,
        "public_id": public_id,
        # Let Cloudinary auto-detect resource_type (image/video/raw)
        "resource_type": "auto",
        "overwrite": False,
        "unique_filename": False,
        "use_filename": True,
    }

    # Check for optional upload preset
    upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    if upload_preset:
        params["upload_preset"] = upload_preset

    resp = cloudinary.uploader.upload(url, **params)
    return {
        "public_id": resp.get("public_id"),
        "secure_url": resp.get("secure_url"),
        "resource_type": resp.get("resource_type"),
        "bytes": resp.get("bytes"),
        "format": resp.get("format"),
        "width": resp.get("width"),
        "height": resp.get("height"),
        "folder": resp.get("folder"),
    }

# ----------------------------
# 3) Do the work
# ----------------------------
def process_folder(dbx: dropbox.Dropbox, shared_url: str, folder_name: str | None = None):
    """Process a single Dropbox shared folder"""
    if not folder_name:
        # Use the last part of the URL path as folder name
        folder_name = urlparse(shared_url).path.rstrip('/').split('/')[-1]
    
    print(f"\nProcessing folder: {folder_name}")
    files = list_files_in_shared_folder(dbx, shared_url)
    media_files = [f for f in files if f[4]]  # is_media
    
    print(f"Found {len(files)} files; {len(media_files)} media candidates.")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare CSV files
    success_file = os.path.join(output_dir, f'uploads_SUCCESS_{timestamp}.csv')
    error_file = os.path.join(output_dir, f'uploads_FAILED_{timestamp}.csv')
    
    with open(success_file, 'w', newline='', encoding='utf-8') as sf, \
         open(error_file, 'w', newline='', encoding='utf-8') as ef:
        
        success_writer = csv.writer(sf)
        error_writer = csv.writer(ef)
        
        success_writer.writerow(['filename', 'cloudinary_url', 'public_id', 'format', 'size'])
        error_writer.writerow(['filename', 'error'])
        
        for display_path, path_lower, name, _id, _is_media in media_files:
            try:
                direct_url = get_direct_file_url(dbx, path_lower)
                result = upload_to_cloudinary(direct_url, folder_name, name)
                success_writer.writerow([
                    name,
                    result["secure_url"],
                    result["public_id"],
                    result["format"],
                    result["bytes"]
                ])
                print(f"✔ Uploaded: {name} → {result['secure_url']}")
            except Exception as e:
                error_writer.writerow([name, str(e)])
                print(f"✖ Failed: {name} — {e}")
            # tiny pause to be gentle with APIs
            time.sleep(0.1)
    
    return success_file, error_file

def main():
    """Main application"""
    import argparse
    parser = argparse.ArgumentParser(description="Upload files from Dropbox shared folder to Cloudinary")
    parser.add_argument("dropbox_url", help="Dropbox shared folder URL")
    parser.add_argument("-n", "--name", help="Custom folder name for uploads (optional)")
    args = parser.parse_args()
    
    try:
        # Initialize services
        init_cloudinary()
        dbx = init_dropbox()
        
        # Process the folder
        success_file, error_file = process_folder(dbx, args.dropbox_url, args.name)
        
        # Print summary
        print("\nUpload Complete!")
        print(f"Success file: {success_file}")
        print(f"Error file: {error_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
