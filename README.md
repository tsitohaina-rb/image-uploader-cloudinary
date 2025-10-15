# Image Uploader to Cloudinary

A comprehensive Python toolkit for uploading images to Cloudinary from various sources: local files, Dropbox, and Google Drive.

## Features

- **Multi-source uploads**: Local files, Dropbox cloud-to-cloud, Google Drive cloud-to-cloud
- **Multi-threaded uploads**: Concurrent processing for improved performance
- **Resumable uploads**: Cache system to resume interrupted uploads
- **Progress tracking**: Real-time progress with success/failure counts
- **Comprehensive logging**: Detailed logs saved to `data/log/`
- **CSV output**: Results saved to `data/output/`
- **Error handling**: Robust error tracking and recovery
- **File validation**: Supports various image formats (.jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .tiff, .tif)

## Prerequisites

- Python 3.x
- Cloudinary account (Get one at [https://cloudinary.com](https://cloudinary.com))

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

Upload images from your local machine to Cloudinary:

```bash
# Upload all images from a folder
python scripts/local_tocloudinary.py /path/to/images

# Upload with custom Cloudinary folder name
python scripts/local_tocloudinary.py /path/to/images my_photos

# Upload single file
python scripts/local_tocloudinary.py /path/to/image.jpg

# Upload with custom folder name and thread count
python scripts/local_tocloudinary.py /path/to/images my_photos 15
```

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

## Output Structure

All scripts generate organized output:

```
data/
├── cache/          # Upload caches for resumable uploads
├── log/           # Detailed execution logs
└── output/        # CSV results with upload URLs
```

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

## Troubleshooting

### Google Drive Setup Issues

**"credentials.json not found"**:

- Download credentials from Google Cloud Console
- Place in root directory with exact filename `credentials.json`

**"Google Drive API not enabled"**:

- Enable Google Drive API in Google Cloud Console

**Permission errors**:

- Ensure you have at least "Viewer" permissions for shared folders

### General Issues

**Python environment**:

- Always activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**Configuration**:

- Verify `.env` file contains correct credentials
- Check `config.py` for upload settings

## Contributing

This toolkit is designed for scalable image management workflows. Each script follows the same pattern for consistency and maintainability.
