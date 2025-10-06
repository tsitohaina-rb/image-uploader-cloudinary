import os
import csv
import sys
import cloudinary.uploader
from pathlib import Path
from config import CLOUDINARY_FOLDER, USE_FILENAME, UNIQUE_FILENAME
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# Thread-safe counter and lock for progress tracking
progress_lock = Lock()
progress_counter = {'uploaded': 0, 'failed': 0, 'total': 0}
error_log = []

def upload_single_image(image_path):
    """
    Upload a single image to Cloudinary.
    
    Args:
        image_path (Path): Path object of the image file
        
    Returns:
        dict: Result dictionary with upload information
    """
    try:
        # Get original extension without the dot
        original_extension = image_path.suffix.lower().replace('.', '')
        
        # Upload to Cloudinary with format parameter to preserve original extension
        response = cloudinary.uploader.upload(
            str(image_path),
            folder=CLOUDINARY_FOLDER,
            use_filename=USE_FILENAME,
            unique_filename=UNIQUE_FILENAME,
            format=original_extension  # Preserve original extension
        )
        
        # Get filename without extension
        filename_without_ext = image_path.stem
        
        result = {
            'local_filename': filename_without_ext,
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
        
    except Exception as e:
        error_message = str(e)
        
        with progress_lock:
            progress_counter['failed'] += 1
            # Log first 10 errors for debugging
            if len(error_log) < 10:
                error_log.append(f"{image_path.name}: {error_message}")
                print(f"ERROR uploading {image_path.name}: {error_message}")
            
            current = progress_counter['uploaded'] + progress_counter['failed']
            if current % 100 == 0 or current == progress_counter['total']:
                print(f"Progress: {current}/{progress_counter['total']} "
                      f"(Success: {progress_counter['uploaded']}, Failed: {progress_counter['failed']})")
        
        # Get filename without extension
        filename_without_ext = image_path.stem
        
        return {
            'local_filename': filename_without_ext,
            'cloudinary_url': 'UPLOAD_FAILED',
            'status': 'failed',
            'error': error_message
        }


def test_cloudinary_connection():
    """Test if Cloudinary is properly configured."""
    try:
        # Check if cloudinary is configured
        if not cloudinary.config().cloud_name:
            return False, "Cloudinary cloud_name is not configured"
        if not cloudinary.config().api_key:
            return False, "Cloudinary api_key is not configured"
        if not cloudinary.config().api_secret:
            return False, "Cloudinary api_secret is not configured"
        
        return True, "Cloudinary configuration looks good"
    except Exception as e:
        return False, f"Configuration error: {str(e)}"


def upload_images_to_cloudinary(image_folder, max_workers=10):
    """
    Upload images from a local folder to Cloudinary using multi-threading and save results to CSV.
    
    Args:
        image_folder (str): Path to folder containing images
        max_workers (int): Number of concurrent upload threads (default: 10, recommended: 5-20)
    """
    
    # Test Cloudinary connection first
    print("Testing Cloudinary configuration...")
    is_configured, message = test_cloudinary_connection()
    print(f"  {message}")
    
    if not is_configured:
        print("\n⚠️  Please check your config.py and .env file and ensure:")
        print("  1. CLOUDINARY_CLOUD_NAME is set correctly")
        print("  2. CLOUDINARY_API_KEY is set correctly")
        print("  3. CLOUDINARY_API_SECRET is set correctly")
        print("\nGet your credentials from: https://cloudinary.com/console")
        return
    
    print("  ✓ Configuration verified\n")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    
    # Get all image files from folder
    image_folder_path = Path(image_folder)
    
    if not image_folder_path.exists():
        print(f"Error: Folder '{image_folder}' does not exist!")
        return
    
    image_files = [
        f for f in image_folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in '{image_folder}'")
        return
    
    # Generate CSV filename based on folder name
    folder_name = image_folder_path.name
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
        future_to_image = {executor.submit(upload_single_image, img): img for img in image_files}
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            result = future.result()
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Write results to CSV (only local_filename and cloudinary_url)
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
        print("Usage: python main.py <folder_path> [max_workers]")
        print("\nExamples:")
        print("  python main.py ./images")
        print("  python main.py ./my_photos 15")
        print("\nArguments:")
        print("  folder_path  : Path to the folder containing images (required)")
        print("  max_workers  : Number of concurrent threads (optional, default: 10)")
        sys.exit(1)
    
    # Get folder path from command line argument
    IMAGE_FOLDER = sys.argv[1]
    
    # Get max_workers from command line argument (optional)
    MAX_WORKERS = 10  # Default value
    if len(sys.argv) >= 3:
        try:
            MAX_WORKERS = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid max_workers value '{sys.argv[2]}', using default: 10")
    
    upload_images_to_cloudinary(IMAGE_FOLDER, max_workers=MAX_WORKERS)