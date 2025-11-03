#!/usr/bin/env python3
"""
Generate CSV output from upload cache file
"""
import json
import csv
import os
import sys
import argparse
import urllib.parse
from datetime import datetime
from pathlib import Path

def is_cloudinary_url(url):
    """Check if URL is a Cloudinary URL."""
    return 'res.cloudinary.com' in url and '/image/upload/' in url

def get_current_format(url):
    """Extract current format from Cloudinary URL."""
    if not is_cloudinary_url(url):
        return None
    
    # Parse the URL to get the filename
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    
    # Get the last part of the path (filename)
    filename = path.split('/')[-1]
    
    # Extract extension
    if '.' in filename:
        return filename.split('.')[-1].lower()
    
    return None

def convert_cloudinary_url_to_jpg(url):
    """
    Convert a Cloudinary URL to JPG format using f_format transformation.
    Returns original URL if already JPG or not a Cloudinary URL.
    """
    if not is_cloudinary_url(url):
        return url  # Return as-is if not a Cloudinary URL
    
    current_format = get_current_format(url)
    
    # Skip if already JPG
    if current_format in ['jpg', 'jpeg']:
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

def generate_csv_from_cache(cache_file_path, output_dir=None):
    """Generate CSV file from cache data"""
    
    # Default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(cache_file_path), "..", "output")
        output_dir = os.path.abspath(output_dir)
    
    # Read cache file
    with open(cache_file_path, 'r') as f:
        cache_data = json.load(f)
    
    successful_uploads = cache_data.get('successful_uploads', {})
    
    if not successful_uploads:
        print("No successful uploads found in cache")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract a meaningful name from cache file or use generic name
    cache_filename = os.path.basename(cache_file_path)
    if "local_upload_cache_" in cache_filename:
        # Try to extract folder name from cache data
        source_path = cache_data.get('source_path', 'unknown')
        folder_name = os.path.basename(source_path)
        output_file = os.path.join(output_dir, f"{folder_name}_upload_results_{timestamp}.csv")
    elif "gdrive_upload_cache_" in cache_filename:
        # Google Drive cache - use folder_path or generate generic name
        folder_path = cache_data.get('folder_path', 'gdrive_folder')
        # Use last part of folder path or generic name
        folder_name = f"gdrive_{folder_path}"
        # Sanitize folder name for filename
        safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('|', '_').replace('?', '_').replace('*', '_').strip()
        output_file = os.path.join(output_dir, f"{safe_folder_name}_upload_results_{timestamp}.csv")
    else:
        output_file = os.path.join(output_dir, f"upload_results_{timestamp}.csv")
    
    # Prepare CSV data
    csv_data = []
    png_converted_count = 0
    
    for file_id_or_path, upload_info in successful_uploads.items():
        # Handle both Google Drive and local upload cache formats
        if 'filename' in upload_info:
            # Google Drive cache format
            filename = upload_info['filename']
            local_path = f"Google Drive File ID: {file_id_or_path}"
            file_size = upload_info.get('file_size', 'N/A')  # Not available in GDrive cache
            public_id = upload_info.get('public_id', 'N/A')
        else:
            # Local upload cache format
            filename = os.path.basename(file_id_or_path)
            local_path = file_id_or_path
            file_size = upload_info.get('file_size', 'N/A')
            public_id = upload_info.get('public_id', 'N/A')
        
        cloudinary_url = upload_info['cloudinary_url']
        upload_time = upload_info['timestamp']
        
        # Convert PNG URLs to JPG
        jpg_url = convert_cloudinary_url_to_jpg(cloudinary_url)
        if jpg_url != cloudinary_url:
            png_converted_count += 1
        
        csv_data.append({
            'filename': filename,
            'local_path': local_path,
            'cloudinary_url': cloudinary_url,
            'jpg_url': jpg_url,  # New column with JPG URLs
            'public_id': public_id,
            'file_size_bytes': file_size,
            'upload_time': upload_time,
            'status': 'SUCCESS'
        })
    
    # Sort by filename for consistency
    csv_data.sort(key=lambda x: x['filename'])
    
    # Write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'local_path', 'cloudinary_url', 'jpg_url', 'public_id', 'file_size_bytes', 'upload_time', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✅ CSV file generated: {output_file}")
    print(f"📊 Total records: {len(csv_data)}")
    print(f"📁 All uploads: SUCCESS")
    print(f"🔄 PNG URLs converted to JPG: {png_converted_count}")
    
    # Display first few URLs to verify they're valid
    print("\n🔗 Sample URLs (first 5):")
    for i, row in enumerate(csv_data[:5]):
        original_format = get_current_format(row['cloudinary_url'])
        jpg_format = get_current_format(row['jpg_url'])
        if original_format != jpg_format:
            format_display = original_format.upper() if original_format else 'UNKNOWN'
            print(f"  {i+1}. {row['filename']} → CONVERTED {format_display} to JPG")
            print(f"     Original: {row['cloudinary_url']}")
            print(f"     JPG:      {row['jpg_url']}")
        else:
            print(f"  {i+1}. {row['filename']} → {row['jpg_url']}")
    
    return output_file

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate CSV output from upload cache file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_csv_from_cache.py path/to/cache.json
  python generate_csv_from_cache.py path/to/cache.json --output /custom/output/dir
  python generate_csv_from_cache.py data/cache/local_upload_cache_*.json
        """
    )
    
    parser.add_argument(
        'cache_file',
        help='Path to the upload cache JSON file'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory for CSV file (default: ../output relative to cache file)'
    )
    
    args = parser.parse_args()
    
    # Validate cache file exists
    if not os.path.exists(args.cache_file):
        print(f"❌ Cache file not found: {args.cache_file}")
        sys.exit(1)
    
    # Generate CSV
    try:
        output_file = generate_csv_from_cache(args.cache_file, args.output_dir)
        print(f"\n✨ Operation completed successfully!")
        print(f"📄 Output file: {output_file}")
    except Exception as e:
        print(f"❌ Error generating CSV: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, show help and use default for backward compatibility
    if len(sys.argv) == 1:
        print("🔄 Running with default cache file for backward compatibility...")
        cache_file = "/Users/tsitohaina/VSCodeProjects/regardbeauty/image-uploader-cloudinary/data/cache/local_upload_cache_ea69da802ee295d191aa85ef82983157.json"
        output_dir = "/Users/tsitohaina/VSCodeProjects/regardbeauty/image-uploader-cloudinary/data/output"
        
        if os.path.exists(cache_file):
            generate_csv_from_cache(cache_file, output_dir)
        else:
            print(f"❌ Default cache file not found: {cache_file}")
            print("\nUsage: python generate_csv_from_cache.py <cache_file> [--output <output_dir>]")
            print("Run with --help for more information")
    else:
        main()