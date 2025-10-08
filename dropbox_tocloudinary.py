import os
import csv
import sys
import cloudinary.uploader
import dropbox
from pathlib import Path
from config import USE_FILENAME, UNIQUE_FILENAME
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import tempfile

# Thread-safe counter and lock for progress tracking
progress_lock = Lock()
progress_counter = {'uploaded': 0, 'failed': 0, 'total': 0}
error_log = []

def upload_single_image_from_dropbox(dbx, dropbox_path, folder_name):
    """
    Download image from Dropbox and upload to Cloudinary.
    
    Args:
        dbx: Dropbox client instance
        dropbox_path (str): Path to file in Dropbox
        folder_name (str): Name of the Dropbox folder (used for Cloudinary folder)
        
    Returns:
        dict: Result dictionary with upload information
    """
    try:
        # Download file from Dropbox to temporary location
        _, response = dbx.files_download(dropbox_path)
        file_data = response.content
        
        # Get filename and extension
        filename = os.path.basename(dropbox_path)
        file_stem = os.path.splitext(filename)[0]
        original_extension = os.path.splitext(filename)[1].lower().replace('.', '')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_data)
            tmp_path = tmp_file.name
        
        try:
            # Upload to Cloudinary with original filename preserved
            response = cloudinary.uploader.upload(
                tmp_path,
                folder=folder_name,
                public_id=file_stem,  # Use original filename without extension
                use_filename=False,    # Don't use the temp filename
                unique_filename=False, # Don't add unique suffix
                overwrite=True,        # Allow overwriting if file exists
                format=original_extension
            )
            
            result = {
                'local_filename': file_stem,
                'cloudinary_url': response['secure_url'],
                'status': 'success'
            }
            
            # Update progress
            with progress_lock:
                progress_counter['uploaded'] += 1
                current = progress_counter['uploaded'] + progress_counter['failed']
                if current % 100 == 0 or current == progress_counter['total']:
                    print(f"Progress: {current}/{progress_counter['total']} "
                          f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']})")
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
    except Exception as e:
        error_message = str(e)
        
        with progress_lock:
            progress_counter['failed'] += 1
            # Log first 10 errors for debugging
            if len(error_log) < 10:
                error_log.append(f"{os.path.basename(dropbox_path)}: {error_message}")
                print(f"ERROR processing {os.path.basename(dropbox_path)}: {error_message}")
            
            current = progress_counter['uploaded'] + progress_counter['failed']
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']})")
        
        filename = os.path.basename(dropbox_path)
        file_stem = os.path.splitext(filename)[0]
        
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
    
    # Generate CSV filename based on folder name
    output_csv = f"{folder_name}.csv"
    
    # Initialize progress counter
    progress_counter['total'] = len(image_files)
    progress_counter['uploaded'] = 0
    progress_counter['failed'] = 0
    error_log.clear()
    
    print(f"Folder: {folder_name}")
    print(f"Found {len(image_files)} images to upload...")
    print(f"Using {max_workers} concurrent threads for faster upload")
    print(f"Output CSV: {output_csv}\n")
    
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for concurrent uploads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_image = {
            executor.submit(upload_single_image_from_dropbox, dbx, img, folder_name): img 
            for img in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            result = future.result()
            results.append(result)
    
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
            
            print(f"\n{'='*60}")
            print(f"✓ Upload completed in {elapsed_time:.2f} seconds")
            if successful > 0:
                print(f"✓ Average speed: {successful/elapsed_time:.2f} images/second")
            print(f"✓ Results saved to '{output_csv}'")
            print(f"  Total images: {len(image_files)}")
            print(f"  Successful uploads: {successful}")
            print(f"  Failed uploads: {failed}")
            
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