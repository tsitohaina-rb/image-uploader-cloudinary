# Cloudinary Image Management Toolkit

A comprehensive Python toolkit for managing images across multiple platforms (Google Drive, Dropbox, Local files) with Cloudinary as the central image hosting solution. This toolkit provides seamless batch uploading, URL management, format conversion, and validation capabilities.

## 🚀 Features

- **Multi-Source Upload Support**: Google Drive, Dropbox, and local file uploads
- **Batch Processing**: CSV-driven bulk uploads from multiple Google Drive folders
- **Multi-threaded Performance**: Concurrent uploads with SSL-safe threading
- **Smart Caching System**: Resume interrupted uploads with timestamp-based cache
- **Format Conversion**: Automated PNG to JPG conversion via Cloudinary transformations
- **URL Management**: Batch download and transform Cloudinary URLs
- **Validation Tools**: Check upload success and identify missing images
- **Progress Tracking**: Real-time upload progress with detailed reporting
- **Error Handling**: Comprehensive retry logic and crash recovery
- **CSV Management**: Advanced CSV processing and consolidation tools
- **Incremental Processing**: Live CSV updates and individual folder backups

## 📋 Prerequisites

- Python 3.8+
- Cloudinary account ([Get one free](https://cloudinary.com))
- Google Drive API credentials (for Google Drive features)
- Dropbox API token (for Dropbox features)

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with your API credentials:

```env
# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Google Drive API (optional)
GOOGLE_DRIVE_CREDENTIALS_FILE=credentials-your-project.json

# Dropbox API (optional)
DROPBOX_TOKEN=your_dropbox_token

# Upload Settings
CLOUDINARY_FOLDER=products
USE_FILENAME=True
UNIQUE_FILENAME=False
```

### 3. Google Drive Setup (Optional)

1. Enable Google Drive API in [Google Cloud Console](https://console.cloud.google.com)
2. Download credentials JSON file
3. Place in project root and update `.env` file

### 4. Dropbox Setup (Optional)

1. Create app in [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Generate access token
3. Add to `.env` file

## 🛠️ Core Scripts

### Main Upload Scripts

#### 1. **Local File Upload** (`main.py`)

Upload images from local directories to Cloudinary.

```bash
# Basic usage
python main.py ./images

# With custom thread count
python main.py ./my_photos 15

# Upload specific folder with 5 workers
python main.py "data/input/product-photos" 5
```

**Features:**

- Multi-threaded uploads (configurable workers)
- Progress tracking with real-time updates
- CSV output with upload results
- Support for all major image formats
- Original filename preservation

#### 2. **Google Drive Upload** (`scripts/googledrive_tocloudinary.py`)

Direct cloud-to-cloud transfer from Google Drive to Cloudinary with advanced features.

```bash
# List available folders and shared items
python scripts/googledrive_tocloudinary.py list
python scripts/googledrive_tocloudinary.py shared
python scripts/googledrive_tocloudinary.py drives

# Upload specific folder
python scripts/googledrive_tocloudinary.py upload FOLDER_ID output_name

# Upload with conservative threading (recommended: 1-5 for stability)
python scripts/googledrive_tocloudinary.py upload FOLDER_ID output_name 1
```

**Features:**

- SSL-safe threading optimized for Google Drive API limitations
- Comprehensive retry logic with exponential backoff
- Timestamp-based cache naming for better organization
- Folder organization preservation with whitespace cleaning
- Memory-efficient streaming uploads with crash recovery
- PNG to JPG URL conversion using Cloudinary transformations
- Support for Shared Drives and team collaboration features

#### 3. **Batch Upload from CSV** (`scripts/batch_upload_from_csv.py`)

Process multiple Google Drive folders from a CSV file with comprehensive batch management.

```bash
# Basic CSV processing
python scripts/batch_upload_from_csv.py data/input/folders.csv

# Custom column names and settings
python scripts/batch_upload_from_csv.py data/input/folders.csv --link-column "Drive Links" --name-column "Product Name"

# Conservative threading for stability
python scripts/batch_upload_from_csv.py data/input/folders.csv --max-workers 1 --no-recursive
```

**Features:**

- CSV-driven batch processing with pandas fallback
- Crash recovery and resume capability
- Incremental CSV generation after each folder
- Individual folder backup caches
- Comprehensive progress tracking and reporting
- Automatic folder validation before processing
- Support for custom folder naming from CSV columns

#### 4. **Dropbox Upload** (`dropbox_tocloudinary.py`)

Upload images from Dropbox folders to Cloudinary.

```bash
# List Dropbox folders
python dropbox_tocloudinary.py list

# Upload specific folder
python dropbox_tocloudinary.py upload /my_images

# Upload with custom thread count
python dropbox_tocloudinary.py upload /photos/vacation 10
```

**Features:**

- Direct Dropbox API integration
- Temporary file handling for efficiency
- Multi-threaded downloads and uploads
- Progress tracking and error logging

#### 5. **Local Upload (Enhanced)** (`scripts/local_tocloudinary.py`)

Advanced local file upload with enhanced features.

```bash
# Upload with enhanced logging
python scripts/local_tocloudinary.py "data/input/brand-photos" 5
```

**Features:**

- Enhanced logging and progress tracking
- Recursive folder scanning
- File type validation
- Detailed upload statistics

### Utility Scripts

#### 6. **Download Cloudinary URLs** (`scripts/download_cloudinary_urls.py`)

Retrieve and optionally convert Cloudinary URLs for folder contents.

```bash
# Download all URLs from a folder
python scripts/download_cloudinary_urls.py products/brand-name

# Convert to JPG format while downloading
python scripts/download_cloudinary_urls.py products/brand-name --format jpg

# Use Search API for better performance on large folders
python scripts/download_cloudinary_urls.py "folder-name" --method search --max-results -1

# Custom output directory
python scripts/download_cloudinary_urls.py products/brand-name --output /custom/path/
```

**Features:**

- Two search methods: Resources API (default) and Search API
- Subfolder support with recursive scanning
- Format conversion using Cloudinary transformations
- CSV export with detailed metadata
- Pagination handling for unlimited image retrieval
- Comprehensive folder discovery

#### 7. **Convert Cloudinary URLs** (`scripts/convert_cloudinary_urls.py`)

Transform existing Cloudinary URLs to different formats via URL modification.

```bash
# Convert URLs in CSV to JPG format
python scripts/convert_cloudinary_urls.py input.csv --format jpg --output output.csv

# Convert to WebP with quality setting
python scripts/convert_cloudinary_urls.py input.csv --format webp --quality 80 --output output.csv

# Specify custom URL column name
python scripts/convert_cloudinary_urls.py input.csv --format png --url-column image_url --output output.csv
```

**Features:**

- URL-based format conversion (no re-upload needed)
- Quality settings for lossy formats (JPG, WebP)
- Batch processing of CSV files
- Support for all Cloudinary formats
- Preserves original URLs alongside converted ones
- Smart format detection and skipping

#### 8. **Check Missing Images** (`scripts/check_missing_images.py`)

Validate Cloudinary URLs and identify failed uploads.

```bash
# Check URLs in CSV file
python scripts/check_missing_images.py results.csv

# Check with custom column name
python scripts/check_missing_images.py results.csv --url-column cloudinary_url
```

**Features:**

- Concurrent URL validation
- Detailed error categorization
- Missing image identification
- Accessibility reporting

### Data Processing Scripts

#### 9. **Consolidate Images** (`scripts/consolidate_images.py`)

Advanced CSV processing for SKU-based image grouping.

```bash
# Process CSV with SKU consolidation
python scripts/consolidate_images.py --input products.csv --output processed.csv
```

#### 10. **Get Images by EAN** (`get_images_by_ean.py`)

Database integration for product image extraction.

```bash
# Interactive EAN search
python get_images_by_ean.py
```

#### 11. **Generate CSV from Cache** (`scripts/generate_csv_from_cache.py`)

Convert upload cache files to CSV format with PNG to JPG conversion.

```bash
# Generate CSV from cache with conversions
python scripts/generate_csv_from_cache.py cache_file.json
```

**Features:**

- Cache file analysis and reporting
- Automatic PNG to JPG URL conversion
- File size and upload time analysis
- Success/failure statistics

## 📁 Project Structure

```
image-uploader-cloudinary/
├── config.py                      # Configuration settings
├── main.py                        # Main local upload script
├── requirements.txt                # Python dependencies
├── credentials-*.json              # Google Drive credentials
├──
├── scripts/                        # Enhanced utility scripts
│   ├── googledrive_tocloudinary.py # Advanced Google Drive uploader
│   ├── batch_upload_from_csv.py    # Batch CSV processing for multiple folders
│   ├── local_tocloudinary.py       # Enhanced local uploader
│   ├── dropbox_to_cloudinary.py    # Alternative Dropbox uploader
│   ├── download_cloudinary_urls.py # URL download/conversion tool
│   ├── convert_cloudinary_urls.py  # URL transformation utility
│   ├── check_missing_images.py     # URL validation tool
│   ├── consolidate_images.py       # CSV processing utility
│   └── generate_csv_from_cache.py  # Cache to CSV converter
├──
├── data/                           # Data directory
│   ├── cache/                      # Upload cache files (JSON)
│   ├── input/                      # Source images and data
│   ├── output/                     # Generated CSV files
│   ├── log/                        # Upload logs
│   └── temp/                       # Temporary files
├──
├── dropbox_tocloudinary.py         # Dropbox upload script
└── get_images_by_ean.py            # Database integration tool
```

## 🔧 Configuration Options

### Cloudinary Settings (`config.py`)

```python
# Upload destination
CLOUDINARY_FOLDER = "products"      # Base folder in Cloudinary

# Filename handling
USE_FILENAME = True                 # Use original filenames
UNIQUE_FILENAME = False             # Don't add unique suffixes

# Performance settings
DEFAULT_MAX_WORKERS = 10            # Default thread count
```

### Environment Variables (`.env`)

```env
# Core Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Optional Service Configurations
GOOGLE_DRIVE_CREDENTIALS_FILE=credentials.json
DROPBOX_TOKEN=your_dropbox_token

# Upload Customization
CLOUDINARY_FOLDER=products
USE_FILENAME=True
UNIQUE_FILENAME=False
```

## 🎯 Common Workflows

### 1. Local to Cloudinary Workflow

```bash
# 1. Prepare images in local folder
mkdir data/input/new-products

# 2. Upload with progress tracking
python main.py "data/input/new-products" 8

# 3. Validate uploads
python scripts/check_missing_images.py new-products.csv

# 4. Convert URLs if needed
python scripts/convert_cloudinary_urls.py new-products.csv converted.csv png jpg
```

### 2. Google Drive to Cloudinary Workflow

```bash
# 1. List available folders
python scripts/googledrive_tocloudinary.py shared

# 2. Upload specific folder (conservative threading for stability)
python scripts/googledrive_tocloudinary.py upload FOLDER_ID brand-name 1

# 3. Download URLs for further processing
python scripts/download_cloudinary_urls.py products/brand-name --format jpg
```

### 3. Batch Upload from CSV Workflow

```bash
# 1. Prepare CSV with Google Drive links
# CSV columns: links, Product Code (optional), description (optional)

# 2. Process batch upload
python scripts/batch_upload_from_csv.py data/input/products.csv --link-column "links" --name-column "Product Code"

# 3. Monitor progress (automatic live updates)
# Check: data/output/batch_results_products_latest.csv

# 4. Resume if interrupted (automatic detection)
python scripts/batch_upload_from_csv.py data/input/products.csv --link-column "links" --name-column "Product Code"
```

### 4. Format Conversion Workflow

```bash
# 1. Get existing URLs
python scripts/download_cloudinary_urls.py products/existing-folder

# 2. Convert PNG to JPG (no re-upload needed)
python scripts/convert_cloudinary_urls.py existing-folder.csv --format jpg --output converted.csv

# 3. Validate converted URLs
python scripts/check_missing_images.py converted.csv
```

## 📊 Output Files

### CSV Structure

All upload scripts generate CSV files with consistent structure:

```csv
local_filename,cloudinary_url
product_001,https://res.cloudinary.com/your-cloud/image/upload/v1.../product_001.jpg
product_002,https://res.cloudinary.com/your-cloud/image/upload/v1.../product_002.jpg
```

### Cache Files

JSON cache files enable resume functionality and progress tracking:

```json
{
	"batch_id": "20241103_142030",
	"csv_file": "data/input/products.csv",
	"started_at": "2024-11-03T14:20:30",
	"processed_folders": [
		{
			"folder_id": "1ABC123...",
			"folder_name": "Brand-Products",
			"status": "success",
			"total_images": 45,
			"successful_uploads": 43,
			"failed_uploads": 1,
			"skipped_uploads": 1,
			"processing_time": 125.5,
			"processed_at": "2024-11-03T14:22:35"
		}
	],
	"failed_folders": [],
	"last_updated": "2024-11-03T14:22:35"
}
```

## 🚨 Troubleshooting

### SSL/Threading Issues (Google Drive)

If experiencing SSL errors with Google Drive uploads:

```bash
# Use single-threaded upload for maximum stability
python scripts/googledrive_tocloudinary.py upload FOLDER_ID output_name 1

# For batch processing, use conservative settings
python scripts/batch_upload_from_csv.py input.csv --max-workers 1

# Alternative: Use local download then upload
python scripts/local_tocloudinary.py "downloaded_folder" 5
```

### Performance Optimization

- **Local uploads**: 5-20 workers (optimal: 8-12)
- **Google Drive**: 1-3 workers (SSL stability critical)
- **Dropbox**: 5-15 workers (network dependent)
- **Batch processing**: 1-5 workers (one per folder)

### Common Error Solutions

1. **"Invalid credentials"**: Check `.env` file configuration
2. **"SSL handshake failed"**: Reduce worker count or use single-threaded
3. **"Rate limit exceeded"**: Add delays between batches
4. **"File not found"**: Verify folder paths and permissions
5. **"Batch interrupted"**: Resume automatically detected on restart
6. **"Column not found"**: Verify CSV column names with `--link-column` option

## 🔄 Cache Management

The toolkit uses timestamp-based caching for resume capability:

```bash
# Cache files location
data/cache/

# Cache naming patterns
{operation}_cache_{folder_name}_{timestamp}.json
batch_cache_{csv_name}_{timestamp}.json
folder_backup_{folder_id}_{batch_id}.json

# Manual cache cleanup (if needed)
rm data/cache/*.json
```

## 📈 Performance Metrics

### Typical Upload Speeds

- **Local files**: 2-10 images/second (network dependent)
- **Google Drive**: 1-5 images/second (API limits)
- **Dropbox**: 3-8 images/second (concurrent downloads)

### Recommended Settings

- **Small batches (<100 images)**: 5-10 workers
- **Large batches (>1000 images)**: 8-15 workers
- **Google Drive uploads**: 1-3 workers (SSL safety)
- **Batch CSV processing**: 1-5 workers per folder

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review error logs in `data/log/`
3. Verify configuration in `.env` and `config.py`
4. Test with smaller batches first

---

**Happy uploading! 🚀**
