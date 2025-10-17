#!/usr/bin/env python3
"""
Convert a remote image URL to another format via Cloudinary and save the result locally.

Modes:
- upload (preferred if you have API credentials): stores a permanent asset.
- fetch  (no credentials needed, only cloud name): does on-the-fly fetch then we download the transformed result.

Supported Input File Types:
- .txt: Plain text files with one URL per line (comments with # are ignored)
- .csv: CSV files with URL column (auto-detects common column names or specify with --url-column)
- .xls/.xlsx: Excel files with URL column (requires pandas and openpyxl: pip install pandas openpyxl)

Supported Output Formats:
- webp, avif, jpg, jpeg, png, gif, bmp, tiff, svg, pdf
- Quality options: auto, eco, 1-100
- Transformations: width, height (with crop=limit to prevent upscaling)

Uses your project's config.py and .env configuration.
"""

import os
import sys
import argparse
import pathlib
import urllib.parse
import requests
import cloudinary
import cloudinary.uploader
import csv
from dotenv import load_dotenv

# Optional imports for Excel support
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup output directory
OUTPUT_DIR = 'data/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_dropbox_url(u: str) -> str:
    # Ensure Dropbox serves raw bytes (not HTML preview)
    try:
        parsed = urllib.parse.urlparse(u)
        if parsed.netloc.endswith("dropbox.com"):
            qs = dict(urllib.parse.parse_qsl(parsed.query))
            if qs.get("raw") != "1":
                qs["raw"] = "1"
            u = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(qs)))
    except Exception:
        pass
    return u

def build_fetch_url(cloud_name: str, src_url: str, fmt: str, width=None, height=None, quality="auto"):
    # Build: https://res.cloudinary.com/<cloud_name>/image/fetch/<transforms>/<encoded_src>
    transforms = [f"f_{fmt}"]
    if quality:
        transforms.append(f"q_{quality}")
    if width:
        transforms.append(f"w_{width}")
    if height:
        transforms.append(f"h_{height}")
    t = ",".join(transforms)
    encoded_src = urllib.parse.quote(src_url, safe="")
    return f"https://res.cloudinary.com/{cloud_name}/image/fetch/{t}/{encoded_src}"

def guess_filename_from_url(u: str, default="download"):
    try:
        path = urllib.parse.urlparse(u).path
        name = pathlib.Path(path).name or default
        # strip extension; we'll add our own based on output format
        stem = pathlib.Path(name).stem or default
        return stem
    except Exception:
        return default

def download_to_file(url: str, out_path: pathlib.Path):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

def read_urls_from_file(file_path: str, url_column: str = None):
    """
    Read URLs from various file formats (txt, csv, xls, xlsx)
    
    Args:
        file_path (str): Path to the file containing URLs
        url_column (str): Column name for CSV/Excel files (optional, will auto-detect)
    
    Returns:
        list: List of URLs
    """
    path_obj = pathlib.Path(file_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = path_obj.suffix.lower()
    urls = []
    
    try:
        if extension == '.txt':
            # Text file - one URL per line
            with open(path_obj, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        elif extension == '.csv':
            # CSV file
            with open(path_obj, 'r', encoding='utf-8') as f:
                # Try to detect if first row is header
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(sample)
                
                reader = csv.reader(f)
                if has_header:
                    headers = next(reader)
                    # Auto-detect URL column
                    if url_column:
                        if url_column not in headers:
                            raise ValueError(f"Column '{url_column}' not found. Available columns: {headers}")
                        url_index = headers.index(url_column)
                    else:
                        # Look for common URL column names
                        url_candidates = ['url', 'urls', 'link', 'links', 'image_url', 'src', 'source']
                        url_index = None
                        for candidate in url_candidates:
                            if candidate.lower() in [h.lower() for h in headers]:
                                url_index = [h.lower() for h in headers].index(candidate.lower())
                                break
                        
                        if url_index is None:
                            # If no obvious column, use first column
                            url_index = 0
                            print(f"⚠️ No URL column specified, using first column: '{headers[0]}'")
                else:
                    # No header, assume first column
                    url_index = 0
                
                for row in reader:
                    if len(row) > url_index and row[url_index].strip():
                        url = row[url_index].strip()
                        if url.startswith(('http://', 'https://')):
                            urls.append(url)
        
        elif extension in ['.xls', '.xlsx']:
            # Excel file
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for Excel file support. Install with: pip install pandas openpyxl")
            
            df = pd.read_excel(path_obj)
            
            if url_column:
                if url_column not in df.columns:
                    raise ValueError(f"Column '{url_column}' not found. Available columns: {list(df.columns)}")
                url_series = df[url_column]
            else:
                # Auto-detect URL column
                url_candidates = ['url', 'urls', 'link', 'links', 'image_url', 'src', 'source']
                url_column_found = None
                
                for candidate in url_candidates:
                    for col in df.columns:
                        if candidate.lower() == str(col).lower():
                            url_column_found = col
                            break
                    if url_column_found:
                        break
                
                if url_column_found:
                    url_series = df[url_column_found]
                    print(f"📊 Using column: '{url_column_found}'")
                else:
                    # Use first column
                    url_series = df.iloc[:, 0]
                    print(f"⚠️ No URL column specified, using first column: '{df.columns[0]}'")
            
            # Filter valid URLs
            urls = [str(url).strip() for url in url_series.dropna() 
                   if str(url).strip().startswith(('http://', 'https://'))]
        
        else:
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: .txt, .csv, .xls, .xlsx")
    
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")
    
    if not urls:
        raise ValueError(f"No valid URLs found in {file_path}")
    
    print(f"📁 Loaded {len(urls)} URLs from {file_path}")
    return urls

def setup_cloudinary():
    """Setup Cloudinary configuration using project's environment"""
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    
    # For fetch mode, we only need cloud_name
    if not cloud_name:
        raise RuntimeError("CLOUDINARY_CLOUD_NAME is required. Check your .env file.")
    
    # Only configure full credentials if upload mode is explicitly used
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    
    return cloud_name

def upload_convert_and_get_url(src_url, fmt, quality, width, height):
    """Upload image to Cloudinary with conversion"""
    
    params = {
        "resource_type": "image",
        "format": fmt,
        "folder": "converted_images",  # Store in a dedicated folder
    }
    if quality and quality != "auto":
        params["quality"] = quality
    if width:
        params["width"] = width
        params["crop"] = "limit"        # avoid upscaling
    if height:
        params["height"] = height
        params["crop"] = params.get("crop", "limit")

    try:
        resp = cloudinary.uploader.upload(src_url, **params)
        return resp.get("secure_url")
    except Exception as e:
        raise RuntimeError(f"Cloudinary upload failed: {e}")

def main():
    p = argparse.ArgumentParser(description="Convert remote images to another format via Cloudinary and save locally.")
    
    # URL input options (mutually exclusive)
    url_group = p.add_mutually_exclusive_group(required=True)
    url_group.add_argument("--url", "-u", help="Single source image URL")
    url_group.add_argument("--urls", "-U", nargs="+", help="Multiple source image URLs (space-separated)")
    url_group.add_argument("--url-file", help="File containing URLs (.txt, .csv, .xls, .xlsx)")
    
    p.add_argument("--url-column", help="Column name for CSV/Excel files (auto-detects if not specified)")
    
    p.add_argument("--format", "-f", required=True, help="Output format (e.g., webp, avif, jpg, png)")
    p.add_argument("--out-dir", "-d", help="Output directory. Default: data/output/")
    p.add_argument("--mode", choices=["fetch", "upload"], default="fetch",
                   help="Conversion mode. 'fetch' (default) does on-the-fly conversion, 'upload' stores permanently.")
    p.add_argument("--quality", "-q", default="auto", help="Quality (e.g., auto, 80, eco)")
    p.add_argument("--width", type=int, help="Optional target width (no upscale).")
    p.add_argument("--height", type=int, help="Optional target height (no upscale).")
    p.add_argument("--max-workers", type=int, default=5, help="Number of concurrent downloads (default: 5)")
    args = p.parse_args()

    print("🔧 Setting up Cloudinary...")
    try:
        cloud_name = setup_cloudinary()
        print(f"✅ Connected to Cloudinary: {cloud_name}")
    except Exception as e:
        print(f"❌ Cloudinary setup failed: {e}")
        sys.exit(1)

    # Collect URLs
    urls = []
    if args.url:
        urls = [args.url]
    elif args.urls:
        urls = args.urls
    elif args.url_file:
        try:
            urls = read_urls_from_file(args.url_file, args.url_column)
        except Exception as e:
            print(f"❌ Error reading URL file: {e}")
            sys.exit(1)

    if not urls:
        print("❌ No URLs provided")
        sys.exit(1)

    # Setup output directory
    output_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format.lower()
    
    print(f"🔄 Converting {len(urls)} image(s) to {fmt}...")
    print(f"📁 Output directory: {output_dir}")

    # Verify upload credentials if needed
    if args.mode == "upload":
        if not (os.getenv("CLOUDINARY_API_KEY") and os.getenv("CLOUDINARY_API_SECRET")):
            print("❌ Upload mode requires CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET")
            sys.exit(1)

    # Process URLs
    if len(urls) == 1:
        # Single URL - use existing logic
        process_single_url(urls[0], fmt, output_dir, cloud_name, args)
    else:
        # Multiple URLs - use concurrent processing
        process_multiple_urls(urls, fmt, output_dir, cloud_name, args)

def process_single_url(url, fmt, output_dir, cloud_name, args):
    """Process a single URL"""
    src = normalize_dropbox_url(url)
    stem = guess_filename_from_url(src)
    out_path = output_dir / f"{stem}.{fmt}"

    print(f"🔍 Processing: {src}")
    
    try:
        if args.mode == "upload":
            print("📤 Using upload mode (permanent storage)...")
            final_url = upload_convert_and_get_url(
                src_url=src, fmt=fmt, quality=args.quality, width=args.width, height=args.height
            )
        else:
            print("🔍 Using fetch mode (on-the-fly conversion, no storage)...")
            final_url = build_fetch_url(
                cloud_name=cloud_name, src_url=src, fmt=fmt,
                width=args.width, height=args.height, quality=args.quality
            )

        print(f"⬇️ Downloading converted image...")
        download_to_file(final_url, out_path)
        print(f"✅ Converted image saved: {out_path.resolve()}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

def process_multiple_urls(urls, fmt, output_dir, cloud_name, args):
    """Process multiple URLs concurrently"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    print(f"🚀 Processing {len(urls)} URLs with {args.max_workers} workers...")
    
    results = {'success': 0, 'failed': 0, 'errors': []}
    start_time = time.time()
    
    def convert_single_url(url_data):
        url, index = url_data
        try:
            src = normalize_dropbox_url(url)
            stem = guess_filename_from_url(src)
            out_path = output_dir / f"{stem}.{fmt}"
            
            if args.mode == "upload":
                final_url = upload_convert_and_get_url(
                    src_url=src, fmt=fmt, quality=args.quality, width=args.width, height=args.height
                )
            else:
                final_url = build_fetch_url(
                    cloud_name=cloud_name, src_url=src, fmt=fmt,
                    width=args.width, height=args.height, quality=args.quality
                )
            
            download_to_file(final_url, out_path)
            return {'success': True, 'url': url, 'output': str(out_path), 'index': index}
            
        except Exception as e:
            return {'success': False, 'url': url, 'error': str(e), 'index': index}
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks with index for ordering
        future_to_url = {
            executor.submit(convert_single_url, (url, i)): url 
            for i, url in enumerate(urls)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            result = future.result()
            
            if result['success']:
                results['success'] += 1
                print(f"✅ [{result['index']+1}/{len(urls)}] {os.path.basename(result['output'])}")
            else:
                results['failed'] += 1
                results['errors'].append(f"{result['url']}: {result['error']}")
                print(f"❌ [{result['index']+1}/{len(urls)}] Failed: {result['url'][:50]}...")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎯 Batch conversion completed in {elapsed_time:.2f} seconds")
    print(f"✅ Successful: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    
    if results['success'] > 0:
        print(f"📁 All files saved to: {output_dir}")
        print(f"⚡ Average speed: {results['success']/elapsed_time:.2f} conversions/second")
    
    if results['errors']:
        print(f"\n⚠️ Errors (first 5):")
        for error in results['errors'][:5]:
            print(f"  - {error}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
