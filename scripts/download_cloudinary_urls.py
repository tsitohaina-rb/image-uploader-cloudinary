#!/usr/bin/env python3
"""
Download/retrieve URLs for all images in a Cloudinary folder
"""
import os
import sys
import csv
import argparse
from datetime import datetime
import cloudinary
import cloudinary.api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def convert_cloudinary_url_format(url, target_format):
    """
    Convert a Cloudinary URL to a different format using f_format transformation
    
    Args:
        url (str): Original Cloudinary URL
        target_format (str): Target format (jpg, png, webp, etc.)
    
    Returns:
        str: Converted URL with format transformation
    """
    if not url or 'cloudinary.com' not in url:
        return url
    
    # Insert the format transformation after /upload/
    if '/upload/' in url:
        parts = url.split('/upload/')
        if len(parts) == 2:
            base_url = parts[0] + '/upload/'
            rest_url = parts[1]
            
            # Add format transformation right after /upload/
            converted_url = f"{base_url}f_{target_format}/{rest_url}"
            
            return converted_url
    
    return url

def setup_cloudinary():
    """Setup Cloudinary configuration"""
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )

def get_images_from_folder(folder_name, max_results=500):
    """
    Retrieve all images from a specific Cloudinary folder
    
    Args:
        folder_name (str): Name of the Cloudinary folder
        max_results (int): Maximum number of results to retrieve
    
    Returns:
        list: List of image resources
    """
    print(f"🔍 Searching for images in folder: {folder_name}")
    
    all_resources = []
    next_cursor = None
    
    try:
        while True:
            # Search for resources in the specific folder
            if next_cursor:
                result = cloudinary.api.resources(
                    type='upload',
                    prefix=folder_name,
                    max_results=min(max_results, 500),  # Cloudinary API limit is 500 per request
                    next_cursor=next_cursor
                )
            else:
                result = cloudinary.api.resources(
                    type='upload',
                    prefix=folder_name,
                    max_results=min(max_results, 500)
                )
            
            resources = result.get('resources', [])
            all_resources.extend(resources)
            
            print(f"📁 Found {len(resources)} images in this batch (Total: {len(all_resources)})")
            
            # Check if there are more results
            next_cursor = result.get('next_cursor')
            if not next_cursor or len(all_resources) >= max_results:
                break
        
        print(f"✅ Total images found: {len(all_resources)}")
        return all_resources[:max_results]  # Limit to max_results
        
    except Exception as e:
        print(f"❌ Error retrieving images: {str(e)}")
        return []

def get_images_from_folder_search(folder_name, max_results=500):
    """Retrieve images using Cloudinary Search API (supports deeper querying).

    Args:
        folder_name (str): Folder to search (exact match).
        max_results (int): Maximum number of resources to return (use -1 or None for all).

    Returns:
        list: List of image resources.
    """
    from cloudinary import Search
    print(f"🔍 (Search API) Searching for images in folder: {folder_name}")

    expression = f'folder="{folder_name}"'
    search = Search().expression(expression).max_results(500)
    all_resources = []
    try:
        data = search.execute()
        batch = data.get('resources', [])
        all_resources.extend(batch)
        print(f"📁 Found {len(batch)} images in first batch (Total: {len(all_resources)})")
        while 'next_cursor' in data:
            if max_results != -1 and len(all_resources) >= max_results:
                break
            cursor = data['next_cursor']
            search = Search().expression(expression).max_results(500).next_cursor(cursor)
            data = search.execute()
            batch = data.get('resources', [])
            all_resources.extend(batch)
            print(f"📁 Found {len(batch)} images in next batch (Total: {len(all_resources)})")
        print(f"✅ Total images found (Search API): {len(all_resources)}")
        return all_resources if max_results == -1 else all_resources[:max_results]
    except Exception as e:
        print(f"❌ Search API error: {e}")
        return []

def export_to_csv(images, folder_name, output_dir="data/output", target_format=None):
    """
    Export image URLs to CSV file
    
    Args:
        images (list): List of image resources from Cloudinary
        folder_name (str): Name of the folder (for filename)
        output_dir (str): Output directory for CSV
        target_format (str): Target format for conversion (jpg, png, webp, etc.)
    
    Returns:
        str: Path to the created CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize folder name for filename (replace slashes and other problematic characters)
    safe_folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{safe_folder_name}_URLs_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Prepare CSV data
    csv_data = []
    
    for img in images:
        filename = os.path.basename(img.get('public_id', ''))
        original_url = img.get('secure_url', '')
        
        # Apply format conversion if target_format is specified
        if target_format:
            # Check if the image is already in the target format by looking at the URL extension
            url_ext = os.path.splitext(original_url)[1].lower().lstrip('.')
            if url_ext == target_format.lower() or (target_format.lower() == 'jpg' and url_ext == 'jpeg'):
                # Keep images that are already in target format unchanged
                final_url = original_url
            else:
                # Convert other formats to target format
                final_url = convert_cloudinary_url_format(original_url, target_format)
        else:
            # No conversion requested, keep original URL
            final_url = original_url
        
        csv_data.append({
            'filename': filename,
            'secure_url': final_url
        })
    
    # Sort by filename for consistency
    csv_data.sort(key=lambda x: x['filename'])
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'secure_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✅ CSV file created: {csv_path}")
    print(f"📊 Total records: {len(csv_data)}")
    
    # Display sample URLs
    print("\n🔗 Sample URLs (first 5):")
    for i, row in enumerate(csv_data[:5]):
        print(f"  {i+1}. {row['filename']} → {row['secure_url']}")
    
    return csv_path

def list_folders(max_pages: int = 50):
    """Comprehensively list all folders in Cloudinary by combining root_folders API and
    paginated scan of resources for public_id prefixes.

    Args:
        max_pages (int): Safety cap on pagination loops to avoid infinite scanning.

    Returns:
        list[str]: Sorted list of discovered folder names.
    """
    discovered = set()
    print("📂 Discovering folders in your Cloudinary account...")

    # Try root_folders (may not be enabled on all accounts / plans)
    try:
        root = cloudinary.api.root_folders()
        for f in root.get('folders', []):
            name = f.get('name')
            if name:
                discovered.add(name)
                print(f"   • root: {name}")
    except Exception as e:
        print(f"   ⚠️  root_folders not available: {e}")

    # Note: sub_folders endpoint not available on all plans; skipping recursive enumeration.

    # Paginated resource scan (public_id prefixes)
    print("🔍 Scanning resources for folder prefixes...")
    next_cursor = None
    page = 0
    try:
        while True:
            page += 1
            if page > max_pages:
                print(f"   ⚠️  Stopping after {max_pages} pages (safety cap)")
                break
            resp = cloudinary.api.resources(
                type='upload',
                max_results=500,
                next_cursor=next_cursor
            ) if next_cursor else cloudinary.api.resources(type='upload', max_results=500)

            resources = resp.get('resources', [])
            print(f"   Page {page}: {len(resources)} resources")
            for r in resources:
                public_id = r.get('public_id', '')
                if '/' in public_id:
                    prefix = public_id.split('/')[0]
                    if prefix:
                        discovered.add(prefix)
            next_cursor = resp.get('next_cursor')
            if not next_cursor:
                break
    except Exception as e:
        print(f"   ⚠️  Resource scan interrupted: {e}")

    folders = sorted(discovered)
    print(f"\n📁 Folder count: {len(folders)}")
    if not folders:
        print("   No folders found")
    else:
        for i, name in enumerate(folders, 1):
            print(f"   {i}. {name}")
    print()
    return folders

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download URLs for all images in a Cloudinary folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_cloudinary_urls.py EURO-Goodwin
  python download_cloudinary_urls.py EURO-Goodwin --output /custom/output/dir
  python download_cloudinary_urls.py --list-folders
  python download_cloudinary_urls.py EURO-Goodwin --max-results 1000
        """
    )
    
    parser.add_argument(
        'folder_name',
        nargs='?',
        help='Name of the Cloudinary folder to download URLs from'
    )
    
    parser.add_argument(
        '--list-folders',
        action='store_true',
        help='List all available folders in Cloudinary'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        default='data/output',
        help='Output directory for CSV file (default: data/output)'
    )
    
    parser.add_argument(
        '--max-results',
        type=int,
        default=500,
        help='Maximum number of images to retrieve (default: 500)'
    )
    parser.add_argument(
        '--method',
        choices=['api', 'search'],
        default='api',
        help='Retrieval method: api (default) or search (Search API)'
    )
    
    parser.add_argument(
        '--format',
        dest='target_format',
        help='Convert URLs to specified format (jpg, png, webp, etc.). Images already in this format will be unchanged.'
    )
    
    args = parser.parse_args()
    
    # Setup Cloudinary
    try:
        setup_cloudinary()
        print("🔧 Cloudinary configured successfully")
    except Exception as e:
        print(f"❌ Error configuring Cloudinary: {str(e)}")
        sys.exit(1)
    
    # List folders if requested
    if args.list_folders:
        list_folders()
        if not args.folder_name:
            return
    
    # Validate folder name
    if not args.folder_name:
        print("❌ Please provide a folder name or use --list-folders to see available folders")
        parser.print_help()
        sys.exit(1)
    
    # Get images from folder
    if args.method == 'search':
        images = get_images_from_folder_search(args.folder_name, args.max_results)
    else:
        images = get_images_from_folder(args.folder_name, args.max_results)
    
    if not images:
        print(f"❌ No images found in folder '{args.folder_name}'")
        print("💡 Use --list-folders to see available folders")
        sys.exit(1)
    
    # Export to CSV
    try:
        csv_path = export_to_csv(images, args.folder_name, args.output_dir, args.target_format)
        print(f"\n✨ Operation completed successfully!")
        print(f"📄 CSV file: {csv_path}")
        if args.target_format:
            print(f"🔄 URLs converted to {args.target_format.upper()} format (except those already in {args.target_format.upper()})")
        print(f"🔗 All image URLs are now available in the CSV file")
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()