#!/usr/bin/env python3
"""
Script to check which Cloudinary URLs from CSV are actually accessible.
This helps identify missing images that were supposed to be uploaded.
"""

import csv
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def check_url_status(row):
    """Check if a Cloudinary URL is accessible"""
    filename = row['local_filename']
    url = row['cloudinary_url']
    folder_path = row.get('folder_path', '')
    
    try:
        # Set a reasonable timeout
        response = requests.head(url, timeout=10, allow_redirects=True)
        
        if response.status_code == 200:
            return {
                'filename': filename,
                'url': url,
                'folder_path': folder_path,
                'status': 'OK',
                'status_code': response.status_code,
                'content_type': response.headers.get('Content-Type', 'unknown')
            }
        else:
            return {
                'filename': filename,
                'url': url,
                'folder_path': folder_path,
                'status': 'ERROR',
                'status_code': response.status_code,
                'error': f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'filename': filename,
            'url': url,
            'folder_path': folder_path,
            'status': 'ERROR',
            'status_code': 'N/A',
            'error': str(e)
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_missing_images.py <csv_file>")
        print("Example: python check_missing_images.py 'data/output/Objets Connectés 10 inkasus DROPSHIP NE PAS EXPEDIER NOEL - Novembre_20251031_142236.csv'")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print(f"🔍 Checking accessibility of URLs in: {csv_file}")
    print("=" * 70)
    
    # Read CSV file
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)
    
    print(f"📊 Found {len(rows)} URLs to check")
    print(f"⏳ Checking accessibility (this may take a moment)...\n")
    
    # Check URLs concurrently for faster processing
    results = []
    accessible_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all URL checks
        future_to_row = {executor.submit(check_url_status, row): row for row in rows}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_row), 1):
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'OK':
                    accessible_count += 1
                    print(f"✅ {result['filename']:<25} - OK ({result['content_type']})")
                else:
                    error_count += 1
                    print(f"❌ {result['filename']:<25} - {result['error']}")
                
                # Show progress every 10 items
                if i % 10 == 0:
                    print(f"\n📈 Progress: {i}/{len(rows)} checked")
                    
            except Exception as e:
                row = future_to_row[future]
                error_count += 1
                print(f"❌ {row['local_filename']:<25} - Exception: {e}")
    
    print("\n" + "=" * 70)
    print("🏁 FINAL RESULTS")
    print("=" * 70)
    print(f"📊 Total URLs checked: {len(rows)}")
    print(f"✅ Accessible images: {accessible_count}")
    print(f"❌ Missing/Error images: {error_count}")
    print(f"📈 Success rate: {(accessible_count/len(rows)*100):.1f}%")
    
    # Show detailed error report
    if error_count > 0:
        print(f"\n⚠️  MISSING/ERROR IMAGES ({error_count} items):")
        print("-" * 70)
        for result in results:
            if result['status'] == 'ERROR':
                folder_info = f" (in {result['folder_path']})" if result['folder_path'] else ""
                print(f"❌ {result['filename']}{folder_info}")
                print(f"   URL: {result['url']}")
                print(f"   Error: {result['error']}")
                print()
        
        print("💡 RECOMMENDATIONS:")
        print("1. Clear cache: rm data/cache/gdrive_upload_cache_*.json")
        print("2. Re-upload missing files:")
        print(f"   python scripts/googledrive_tocloudinary.py upload 1oRheMGQzP47fW2uMrVfhKAtG5d5T4CxW \"Objets Connectés 10 inkasus DROPSHIP NE PAS EXPEDIER NOEL - Novembre\" 1")
        print("3. Or use --force flag when implemented")
    
    else:
        print("🎉 All images are accessible on Cloudinary!")
    
    print("=" * 70)

if __name__ == "__main__":
    main()