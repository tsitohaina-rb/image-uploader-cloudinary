#!/usr/bin/env python3
"""
Generate CSV output from upload cache file
"""
import json
import csv
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

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
    else:
        output_file = os.path.join(output_dir, f"upload_results_{timestamp}.csv")
    
    # Prepare CSV data
    csv_data = []
    
    for local_path, upload_info in successful_uploads.items():
        filename = os.path.basename(local_path)
        cloudinary_url = upload_info['cloudinary_url']
        file_size = upload_info['file_size']
        upload_time = upload_info['timestamp']
        
        csv_data.append({
            'filename': filename,
            'local_path': local_path,
            'cloudinary_url': cloudinary_url,
            'file_size_bytes': file_size,
            'upload_time': upload_time,
            'status': 'SUCCESS'
        })
    
    # Sort by filename for consistency
    csv_data.sort(key=lambda x: x['filename'])
    
    # Write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'local_path', 'cloudinary_url', 'file_size_bytes', 'upload_time', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✅ CSV file generated: {output_file}")
    print(f"📊 Total records: {len(csv_data)}")
    print(f"📁 All uploads: SUCCESS")
    
    # Display first few URLs to verify they're valid
    print("\n🔗 Sample URLs (first 5):")
    for i, row in enumerate(csv_data[:5]):
        print(f"  {i+1}. {row['filename']} → {row['cloudinary_url']}")
    
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