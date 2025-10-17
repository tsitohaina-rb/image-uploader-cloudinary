# Image Management Toolkit for Cloudinary

A comprehensive Python toolkit for image management with Cloudinary integration. Upload, download, convert, and manage images from various sources with powerful automation features.

## Features

- **Multi-source uploads**: Local files, Dropbox cloud-to-cloud, Google Drive cloud-to-cloud
- **Image conversion**: Convert remote images between formats via Cloudinary (WebP, AVIF, JPG, PNG, etc.)
- **URL extraction**: Download image URLs from Cloudinary folders with pagination support
- **Multi-threaded processing**: Concurrent operations for improved performance
- **Resumable uploads**: Cache system to resume interrupted uploads
- **Batch processing**: Handle multiple files/URLs with progress tracking
- **Multiple input formats**: Support for .txt, .csv, .xls, .xlsx file inputs
- **Progress tracking**: Real-time progress with success/failure counts
- **Comprehensive logging**: Detailed logs saved to `data/log/`
- **CSV output**: Results saved to `data/output/`
- **Error handling**: Robust error tracking and recovery
- **File validation**: Supports various image formats (.jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .tiff, .tif)

## Prerequisites

- Python 3.x
- Cloudinary account (Get one at [https://cloudinary.com](https://cloudinary.com))
- Optional: `pandas` and `openpyxl` for Excel file support (`pip install pandas openpyxl`)

## Setup

1. Clone or download this repository to your local machine.

2. Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   # .venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory with your Cloudinary credentials:

   ```
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```

   Replace `your_cloud_name`, `your_api_key`, and `your_api_secret` with your actual Cloudinary credentials from your Cloudinary Console.

5. (Optional) Configure upload settings in `config.py`:
   - `CLOUDINARY_FOLDER`: The folder name in Cloudinary where images will be uploaded
   - `USE_FILENAME`: Whether to use the original filename (True/False)
   - `UNIQUE_FILENAME`: Whether to add a unique suffix to avoid duplicates (True/False)

## Usage

### 1. Local File Upload

Upload images from your local machine to Cloudinary with full subfolder support:

```bash
# Upload all images from a folder (recursively includes subfolders)
python scripts/local_tocloudinary.py /path/to/images

# Upload with custom Cloudinary folder name
python scripts/local_tocloudinary.py /path/to/images my_photos

# Upload single file
python scripts/local_tocloudinary.py /path/to/image.jpg

# Upload with custom folder name and thread count
python scripts/local_tocloudinary.py /path/to/images my_photos 15
```

**Features**:

- Recursive folder scanning (preserves subfolder structure in Cloudinary)
- Caching system prevents duplicate uploads
- Progress tracking with concurrent uploads
- CSV output with filename and URL mapping

### 2. Dropbox to Cloudinary

Upload images directly from Dropbox to Cloudinary (cloud-to-cloud):

#### Setup

Add your Dropbox token to `.env`:

```
DROPBOX_TOKEN=your_dropbox_token_here
```

#### Usage

```bash
# List all folders in your Dropbox
python scripts/dropbox_tocloudinary.py list

# Upload images from a Dropbox folder
python scripts/dropbox_tocloudinary.py upload /my_images

# Upload with custom thread count
python scripts/dropbox_tocloudinary.py upload /photos/vacation 15
```

### 3. Google Drive to Cloudinary

Upload images directly from Google Drive to Cloudinary (cloud-to-cloud):

#### Setup

1. **Create Google Cloud Project** (if you don't have one):

   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Google Drive API**:

   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it

3. **Create OAuth 2.0 Credentials**:

   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Configure OAuth consent screen if prompted
   - Choose "Desktop application" as application type
   - Download the credentials file as `credentials.json`

4. **Place credentials file**:
   ```
   image-uploader-cloudinary/
   ├── credentials.json  ← Place it here
   ├── config.py
   └── ...
   ```

#### Usage

```bash
# List all folders in your Google Drive
python scripts/googledrive_tocloudinary.py list

# Upload images from a Google Drive folder (using folder ID)
python scripts/googledrive_tocloudinary.py upload 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE

# With custom folder name for Cloudinary
python scripts/googledrive_tocloudinary.py upload FOLDER_ID my_photos

# With custom folder name and thread count
python scripts/googledrive_tocloudinary.py upload FOLDER_ID my_photos 15
```

**Finding Google Drive Folder ID**:

- From URL: `https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE`
- The folder ID is: `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs25kzpKiCkVyiE`
- Or use the `list` command to see all folders with their IDs

### 4. Image Format Conversion

Convert remote images between formats using Cloudinary's transformation API:

#### Single URL Conversion

```bash
# Convert a single image to WebP
python scripts/convert_via_cloudinary.py --url "https://example.com/image.jpg" --format webp

# Convert with quality and size optimization
python scripts/convert_via_cloudinary.py --url "https://example.com/image.jpg" --format webp --quality 80 --width 800
```

#### Batch Conversion from Multiple URLs

```bash
# Convert multiple URLs (space-separated)
python scripts/convert_via_cloudinary.py --urls "https://site1.com/img1.jpg" "https://site2.com/img2.png" --format webp

# With custom output directory and workers
python scripts/convert_via_cloudinary.py --urls url1 url2 url3 --format webp --out-dir "converted/" --max-workers 10
```

#### File-Based Batch Conversion

**Text File (.txt)**:

```txt
# urls.txt
https://example.com/image1.jpg
https://example.com/image2.png
# Comments are ignored
https://example.com/image3.webp
```

**CSV File (.csv)**:

```csv
url,description,category
https://example.com/image1.jpg,Product A,Category 1
https://example.com/image2.png,Product B,Category 2
```

**Excel File (.xlsx)**:
Supports Excel files with URL columns.

**Usage**:

```bash
# Auto-detect URL column in CSV/Excel
python scripts/convert_via_cloudinary.py --url-file urls.txt --format webp
python scripts/convert_via_cloudinary.py --url-file products.csv --format jpg
python scripts/convert_via_cloudinary.py --url-file inventory.xlsx --format png

# Specify custom URL column
python scripts/convert_via_cloudinary.py --url-file data.csv --url-column "image_links" --format webp
```

**Supported Input File Types**:

- `.txt`: Plain text files with one URL per line
- `.csv`: CSV files with URL column (auto-detects: url, urls, link, image_url, src, source)
- `.xls/.xlsx`: Excel files (requires: `pip install pandas openpyxl`)

**Supported Output Formats**:

- Image formats: webp, avif, jpg, jpeg, png, gif, bmp, tiff, svg, pdf
- Quality options: auto, eco, 1-100
- Transformations: width, height (prevents upscaling)

**Conversion Modes**:

- **fetch** (default): On-the-fly conversion, no permanent storage in Cloudinary
- **upload**: Permanent storage in Cloudinary's "converted_images" folder

### 5. Download Cloudinary URLs

Extract image URLs from existing Cloudinary folders:

#### List All Folders

```bash
# Discover all folders in your Cloudinary account
python scripts/download_cloudinary_urls.py --list-folders
```

#### Download URLs from Specific Folder

```bash
# Get all image URLs from a folder (default 500 limit)
python scripts/download_cloudinary_urls.py EURO-Goodwin

# Get ALL images (unlimited)
python scripts/download_cloudinary_urls.py Laguiole --method search --max-results -1

# Use Search API for better performance on large folders
python scripts/download_cloudinary_urls.py "La Peaulie" --method search --max-results -1

# Custom output directory
python scripts/download_cloudinary_urls.py EURO-Goodwin --output /custom/path/
```

**Methods**:

- **api** (default): Uses resources API with prefix filtering
- **search**: Uses Search API for exact folder matching (recommended for large folders)

**Features**:

- Comprehensive folder discovery (scans all resources to find folders)
- Pagination support for unlimited image retrieval
- CSV output with filename and secure_url columns
- Subfolder support (e.g., `python scripts/download_cloudinary_urls.py Folder/Subfolder`)

### 6. Cache Management

Generate CSV from upload cache files:

```bash
# Convert upload cache to CSV
python scripts/generate_csv_from_cache.py
```

**Features**:

- Extracts successful uploads from cache
- Outputs filename, local_path, cloudinary_url, file_size, upload_time, status
- Useful for auditing and reporting

## Additional Tools

### Image Consolidation Script

Consolidate images across product variants based on SKU patterns:

```bash
python scripts/consolidate_images.py --input path/to/your/file.csv
```

This script groups rows by base SKU (ignoring variant indicators) and copies image URLs across all variants of the same product.

### EAN Image Extraction

Extract product images from MySQL database by EAN codes:

```bash
python scripts/get_images_by_ean.py
```

Interactive script for extracting product images by EAN codes with various search options.

## Output Structure

All scripts generate organized output:

```
data/
├── cache/          # Upload caches for resumable uploads & auth tokens
├── input/          # Input files for testing (txt, csv, xlsx)
├── log/            # Detailed execution logs with timestamps
└── output/         # CSV results and converted images
```

**File Naming Conventions**:

- Upload results: `{folder_name}_{timestamp}.csv`
- Converted images: `{original_name}.{new_format}`
- URL exports: `{folder_name}_URLs_{timestamp}.csv`
- Logs: `{operation}_{timestamp}.log`

## Authentication & Security

### Dropbox

- Requires `DROPBOX_TOKEN` in `.env`
- Get token from [Dropbox Developers](https://www.dropbox.com/developers/apps)

### Google Drive

- Requires `credentials.json` in root directory
- First-time authentication opens browser for OAuth
- Tokens cached in `data/cache/token.pickle`
- Only requests read-only access to Google Drive

### Cloudinary

- Requires cloud name, API key, and API secret in `.env`
- All uploads respect configured folder structure

## Error Handling & Recovery

- **Resumable uploads**: All scripts cache successful uploads to avoid duplicates
- **Error logging**: Detailed error tracking with file-specific failure reasons
- **Progress tracking**: Real-time updates on upload status
- **Validation**: Pre-upload checks for connectivity and file accessibility

## Script Overview

| Script                        | Purpose                                | Input                | Output           |
| ----------------------------- | -------------------------------------- | -------------------- | ---------------- |
| `local_tocloudinary.py`       | Upload local images to Cloudinary      | Local files/folders  | CSV with URLs    |
| `dropbox_tocloudinary.py`     | Upload from Dropbox to Cloudinary      | Dropbox paths        | CSV with URLs    |
| `googledrive_tocloudinary.py` | Upload from Google Drive to Cloudinary | Drive folder IDs     | CSV with URLs    |
| `convert_via_cloudinary.py`   | Convert remote images between formats  | URLs, txt, csv, xlsx | Converted images |
| `download_cloudinary_urls.py` | Extract URLs from Cloudinary folders   | Folder names         | CSV with URLs    |
| `generate_csv_from_cache.py`  | Export upload cache to CSV             | Cache files          | CSV reports      |
| `consolidate_images.py`       | Merge images across product variants   | CSV files            | Consolidated CSV |
| `get_images_by_ean.py`        | Extract images by EAN from database    | EAN codes            | CSV with matches |

## Troubleshooting

### Google Drive Setup Issues

**"credentials.json not found"**:

- Download credentials from Google Cloud Console
- Place in root directory with exact filename `credentials.json`

**"Google Drive API not enabled"**:

- Enable Google Drive API in Google Cloud Console

**Permission errors**:

- Ensure you have at least "Viewer" permissions for shared folders

### Excel File Support Issues

**"pandas is required for Excel file support"**:

```bash
pip install pandas openpyxl
```

**Excel file not recognized**:

- Ensure file extension is `.xls` or `.xlsx`
- Check file is not corrupted

### Cloudinary Issues

**"CLOUDINARY_CLOUD_NAME is required"**:

- Check `.env` file exists and contains correct credentials
- Verify environment variables are loaded

**"No images found in folder"**:

- Use `--list-folders` to see available folders
- Check folder name spelling (case-sensitive)
- For subfolders, use format: `ParentFolder/SubFolder`

### General Issues

**Python environment**:

- Always activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**File path issues**:

- Use absolute paths when possible
- Quote paths with spaces: `"path with spaces"`
- Check file permissions and existence

**Performance issues**:

- Reduce `--max-workers` for slower connections
- Use `--method search` for large Cloudinary folders
- Check internet connection stability

## Contributing

This toolkit is designed for scalable image management workflows. Each script follows consistent patterns:

- Environment-based configuration
- Progress tracking and logging
- CSV output for integration
- Error handling and recovery
- Concurrent processing where beneficial

Feel free to extend functionality while maintaining these patterns.
