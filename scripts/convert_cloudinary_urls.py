#!/usr/bin/env python3
"""
Convert Cloudinary URLs to a specified format and output CSV with converted URLs.

This script:
1. Reads a CSV file with Cloudinary URLs
2. Converts URLs to the specified format (e.g., jpg, png, webp)
3. Skips URLs that are already in the target format
4. Outputs a new CSV with the original data plus a new column with converted URLs
5. Maintains the exact order of the input CSV

Usage:
    python convert_cloudinary_urls.py input.csv --format jpg --output output.csv
    python convert_cloudinary_urls.py input.csv --format webp --url-column cloudinary_url --quality 80

Features:
- Preserves all original columns
- Adds a new column with converted URLs
- Skips conversion if already in target format
- Supports quality settings for lossy formats
- Fast processing (URL transformation only, no actual download/upload)
"""

import csv
import argparse
import os
import sys
import urllib.parse
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


def convert_cloudinary_url(url, target_format, quality=None):
    """
    Convert a Cloudinary URL to the specified format using f_format transformation.
    
    Args:
        url (str): Original Cloudinary URL
        target_format (str): Target format (jpg, png, webp, etc.)
        quality (int, optional): Quality setting for lossy formats
    
    Returns:
        str: Converted URL with f_format transformation or original URL if already in target format
    """
    if not is_cloudinary_url(url):
        return url  # Return as-is if not a Cloudinary URL
    
    current_format = get_current_format(url)
    
    # Skip if already in target format
    if current_format == target_format.lower():
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
            
            # Rebuild the path (keeping original filename unchanged)
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


def process_csv(input_file, target_format, url_column, output_file, quality=None):
    """
    Process CSV file and convert Cloudinary URLs.
    
    Args:
        input_file (str): Path to input CSV file
        target_format (str): Target format for conversion
        url_column (str): Name of the column containing URLs
        output_file (str): Path to output CSV file
        quality (int, optional): Quality setting for lossy formats
    """
    
    # Read input CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    if not rows:
        print("❌ No data found in CSV file")
        return
    
    if not fieldnames:
        print("❌ No column headers found in CSV file")
        return
    
    if url_column not in fieldnames:
        print(f"❌ Column '{url_column}' not found in CSV. Available columns: {', '.join(fieldnames)}")
        return
    
    # Add new column name for converted URLs
    converted_column = f'{target_format}_url'
    new_fieldnames = list(fieldnames) + [converted_column]
    
    # Process rows
    converted_count = 0
    skipped_count = 0
    
    print(f"🔄 Processing {len(rows)} rows...")
    print(f"📋 Converting URLs in column '{url_column}' to {target_format.upper()} format")
    if quality:
        print(f"🎨 Using quality: {quality}")
    
    for i, row in enumerate(rows, 1):
        original_url = row[url_column]
        
        if not original_url:
            row[converted_column] = ''
            continue
        
        converted_url = convert_cloudinary_url(original_url, target_format, quality)
        row[converted_column] = converted_url
        
        if converted_url != original_url:
            converted_count += 1
            print(f"✅ [{i}/{len(rows)}] Converted: {os.path.basename(original_url)} → {target_format}")
        else:
            skipped_count += 1
            if i <= 5:  # Show first few skips
                current_format = get_current_format(original_url)
                if current_format == target_format.lower():
                    print(f"⏭️  [{i}/{len(rows)}] Skipped (already {target_format}): {os.path.basename(original_url)}")
                else:
                    print(f"⏭️  [{i}/{len(rows)}] Skipped (not Cloudinary): {original_url[:50]}...")
    
    # Write output CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Processing completed!")
    print(f"📊 Total rows processed: {len(rows)}")
    print(f"🔄 URLs converted: {converted_count}")
    print(f"⏭️  URLs skipped: {skipped_count}")
    print(f"📁 Output saved to: {output_file}")
    print(f"📋 New column added: '{converted_column}'")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cloudinary URLs in CSV to specified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all URLs to JPG format
    python convert_cloudinary_urls.py input.csv --format jpg --output output.csv
    
    # Convert to WebP with quality 80
    python convert_cloudinary_urls.py input.csv --format webp --quality 80 --output output.csv
    
    # Specify custom URL column name
    python convert_cloudinary_urls.py input.csv --format png --url-column image_url --output output.csv

Supported formats: jpg, jpeg, png, webp, avif, gif, bmp, tiff, svg, pdf
        """
    )
    
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--format', '-f', required=True, 
                       help='Target format (jpg, png, webp, etc.)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file path')
    parser.add_argument('--url-column', default='cloudinary_url',
                       help='Name of column containing URLs (default: cloudinary_url)')
    parser.add_argument('--quality', '-q', type=int,
                       help='Quality setting for lossy formats (1-100)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Validate format
    supported_formats = ['jpg', 'jpeg', 'png', 'webp', 'avif', 'gif', 'bmp', 'tiff', 'svg', 'pdf']
    if args.format.lower() not in supported_formats:
        print(f"❌ Unsupported format: {args.format}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    # Validate quality
    if args.quality and (args.quality < 1 or args.quality > 100):
        print("❌ Quality must be between 1 and 100")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        process_csv(
            args.input_file,
            args.format.lower(),
            args.url_column,
            args.output,
            args.quality
        )
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()