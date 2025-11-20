# Cloud Storage to Cloudinary Upload Script

## Overview

Unified enterprise-grade script for uploading images from **Google Drive** or **Dropbox** to **Cloudinary** with advanced multiprocessing, caching, and format conversion capabilities.

## ✨ Key Features

### 🚀 **Performance & Reliability**

- **Multiprocessing Architecture**: Parallel uploads with configurable worker processes
- **Smart Caching System**: Resume functionality with atomic writes and corruption recovery
- **Cross-Platform Support**: Windows, macOS, Linux with proper file locking
- **Connection Pooling**: Optimized HTTP sessions to prevent pool saturation
- **Retry Logic**: Intelligent retry with exponential backoff (auto/true/false modes)

### 📁 **Cloud Storage Integration**

- **Google Drive**: Full API integration with shared drives, folder validation
- **Dropbox**: Complete folder scanning and file access
- **Unified Interface**: Single script for both providers
- **Recursive Scanning**: Optional deep folder traversal

### 🎨 **Image Processing**

- **Automatic Compression**: Smart compression for images >20MB
- **Format Conversion**: Convert to JPG, PNG, WebP, AVIF during upload
- **Quality Control**: Configurable quality settings (1-100)
- **Size Optimization**: Dimension scaling when needed

### 📊 **Batch Operations**

- **CSV Batch Upload**: Process multiple folders from CSV files
- **Consolidated Output**: Single CSV, cache, and log files for batches
- **Progress Tracking**: Real-time progress bars with statistics
- **Comprehensive Reporting**: Detailed Excel output with multiple sheets

### 🔧 **Management & Analysis**

- **Folder Listing**: Browse Google Drive, Dropbox, and Cloudinary folders
- **Shared Content**: Access Google Drive shared files and Team Drives
- **Search Capabilities**: Find specific Cloudinary folders
- **Connection Testing**: Verify all service configurations

## 📋 Requirements

### Python Dependencies

```bash
pip install cloudinary google-api-python-client google-auth-oauthlib dropbox
pip install requests tqdm pillow python-dotenv pathlib
```

### Required Files

- `credentials-bzc.json` - Google Drive API credentials
- `.env` - Cloudinary and Dropbox configuration
- `config.py` - Upload behavior settings

### Environment Variables (.env)

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
DROPBOX_TOKEN=your_dropbox_token
```

## 🚀 Usage

### **List Folders**

```bash
# Google Drive folders
python cloudstorage_tocloudinary.py list gdrive

# Dropbox folders
python cloudstorage_tocloudinary.py list dropbox

# Google Drive shared content
python cloudstorage_tocloudinary.py shared

# Google Drive Team Drives
python cloudstorage_tocloudinary.py drives

# Cloudinary folders
python cloudstorage_tocloudinary.py cloudinary
python cloudstorage_tocloudinary.py cloudinary --folder "product_photos"
```

### **Single Folder Upload**

```bash
# Basic Google Drive upload
python cloudstorage_tocloudinary.py upload gdrive "1ABC123def456"

# With custom folder name and format conversion
python cloudstorage_tocloudinary.py upload gdrive "folder_id" --folder "Products" --format webp --quality 80

# Dropbox upload with retry settings
python cloudstorage_tocloudinary.py upload dropbox "/vacation_photos" --retry true --workers 4

# URL format support
python cloudstorage_tocloudinary.py upload gdrive "https://drive.google.com/drive/folders/1ABC123"
```

### **Batch Upload from CSV**

```bash
# Basic batch upload
python cloudstorage_tocloudinary.py batch-upload folders.csv --provider gdrive

# With format conversion and quality
python cloudstorage_tocloudinary.py batch-upload products.csv --provider gdrive --format webp --quality 85

# Dropbox batch with retry disabled
python cloudstorage_tocloudinary.py batch-upload dropbox_folders.csv --provider dropbox --retry false
```

#### **CSV Format for Batch Upload**

```csv
folder_name,link
Product_A,1ABC123def456
Product_B,https://drive.google.com/drive/folders/1DEF456ghi789
Product_C,/dropbox/folder/path
```

### **Shorthand Commands** (Backwards Compatibility)

```bash
# Google Drive shorthand
python cloudstorage_tocloudinary.py gdrive "folder_id" "Custom_Name" 4

# Dropbox shorthand
python cloudstorage_tocloudinary.py dropbox "/folder/path" "Album_Name" 6
```

## ⚙️ Command Options

### **Upload Options**

- `--folder NAME` - Custom Cloudinary folder name
- `--workers N` - Number of worker processes (default: CPU count)
- `--no-recursive` - Disable recursive folder scanning
- `--force-rescan` - Ignore cache, rescan everything

### **Retry Options**

- `--retry auto` - Retry if <50% uploads failed (default)
- `--retry true` - Always retry failed uploads
- `--retry false` - Never retry failed uploads

### **Format Conversion Options**

- `--format jpg|png|webp|avif` - Convert images to specified format
- `--quality 1-100` - Quality setting for lossy formats (JPG, WebP)

### **Search Options**

- `--folder NAME` - Search Cloudinary for specific folder name

## 📊 Output Files

### **Single Upload**

- **CSV**: `data/output/gdrive_FolderName_20241114_123456.csv`
- **Cache**: `data/cache/gdrive_cache_FolderName_20241114.json`
- **Logs**: `data/log/FolderName_20241114_123456.log`

### **Batch Upload** (Consolidated)

- **CSV**: `data/output/batch_gdrive_20241114_123456.csv`
- **Cache**: `data/cache/batch_gdrive_20241114_123456_cache.json`
- **Logs**: `data/log/batch_gdrive_20241114_123456.log`

### **CSV Output Columns**

- `local_filename` - Original file name
- `cloudinary_url` - Main Cloudinary URL
- `jpg_url` - Converted format URL
- `status` - Upload status (success/failed/skipped)
- `provider` - Source provider (gdrive/dropbox)
- `was_compressed` - Compression applied (true/false)
- `final_size_mb` - Final file size
- `compression_ratio` - Compression efficiency
- `batch_folder` - Folder name for batch uploads
- `target_format` - Requested format
- `quality` - Quality setting used

## 🔧 Advanced Features

### **Compression Logic**

- **Automatic**: Images >20MB are automatically compressed
- **Progressive JPEG**: Uses progressive compression for better results
- **Quality Reduction**: Starts at 85% quality, reduces if needed
- **Dimension Scaling**: Reduces dimensions if quality reduction insufficient
- **Format Optimization**: Maintains optimal format for each image

### **Caching System**

- **Atomic Writes**: Prevents cache corruption during concurrent access
- **Cross-Platform Locking**: Uses fcntl (Unix) or msvcrt (Windows)
- **Resume Capability**: Continue interrupted uploads seamlessly
- **Smart Detection**: Automatically finds and loads recent cache files
- **Corruption Recovery**: Handles and repairs corrupted cache files

### **Error Handling**

- **Connection Recovery**: Automatic retry with exponential backoff
- **SSL Stability**: Optimized for multiprocessing SSL connections
- **Graceful Degradation**: Continues processing even with partial failures
- **Detailed Logging**: Process-specific logs with full error context

### **Platform Compatibility**

- **Windows**: Full Unicode support with proper console encoding
- **macOS/Linux**: Native emoji and symbol support in terminal
- **File Paths**: Cross-platform path handling with pathlib
- **Encoding**: UTF-8 support for international filenames

## 🎯 Use Cases

### **E-commerce Product Management**

```bash
# Upload product images with WebP conversion for web optimization
python cloudstorage_tocloudinary.py batch-upload products.csv --format webp --quality 80

# High-quality images for print catalogs
python cloudstorage_tocloudinary.py upload gdrive "catalog_folder" --format jpg --quality 95
```

### **Media Asset Migration**

```bash
# Migrate large media libraries with compression
python cloudstorage_tocloudinary.py upload gdrive "media_archive" --workers 8 --retry true

# Batch migrate from multiple sources
python cloudstorage_tocloudinary.py batch-upload media_sources.csv --provider gdrive
```

### **Team Collaboration**

```bash
# Process shared Google Drive folders
python cloudstorage_tocloudinary.py shared  # List shared content
python cloudstorage_tocloudinary.py upload gdrive "shared_folder_id" --folder "Team_Photos"

# Handle Team Drives
python cloudstorage_tocloudinary.py drives  # List Team Drives
```

## 📈 Performance Tips

### **Optimize Worker Count**

- **CPU-bound**: Use CPU count (default)
- **Network-bound**: Use 2x CPU count
- **Large files**: Use fewer workers (2-4)
- **Small files**: Use more workers (8-12)

### **Batch Processing**

- **Group similar folders**: Process similar content together
- **Use consolidation**: Let batch upload create single output files
- **Monitor progress**: Use progress bars to track large batches

### **Format Strategy**

- **WebP**: Best for web display (smaller files, good quality)
- **JPG**: Universal compatibility, good compression
- **PNG**: Lossless, best for graphics with transparency
- **AVIF**: Next-gen format, excellent compression (limited support)

## 🚨 Troubleshooting

### **Common Issues**

#### "Connection pool is full" Warning

```bash
# Reduce workers to prevent connection saturation
python cloudstorage_tocloudinary.py upload gdrive "folder_id" --workers 2
```

#### SSL/Certificate Errors

```bash
# Force rescan to clear corrupted cache
python cloudstorage_tocloudinary.py upload gdrive "folder_id" --force-rescan
```

#### Large File Upload Failures

```bash
# The script automatically compresses >20MB files
# Check logs for compression details
cat data/log/FolderName_*.log | grep -i compress
```

#### Cache Corruption

- Script automatically detects and handles corruption
- Prompts for manual fixing when needed
- Use `--force-rescan` to start fresh

### **Performance Issues**

- **Slow uploads**: Reduce worker count or check network
- **High memory usage**: Process smaller batches
- **Frequent timeouts**: Check network stability

## 🔗 Integration

### **With Other Scripts**

```python
# Import and use directly in your scripts
from cloudstorage_tocloudinary import upload_cloud_folder_to_cloudinary

results = upload_cloud_folder_to_cloudinary(
    source_path="folder_id",
    provider="gdrive",
    folder_name="Custom_Name",
    max_workers=4,
    target_format="webp",
    quality=80
)
```

### **Automation Examples**

```bash
#!/bin/bash
# Daily product image sync
python cloudstorage_tocloudinary.py batch-upload daily_products.csv --format webp --quality 85

# Weekly media backup with high quality
python cloudstorage_tocloudinary.py upload gdrive "backup_folder" --format jpg --quality 95 --retry true
```

## 📚 Related Scripts

- `url_tocloudinary.py` - Upload from direct URLs with 404 detection
- `batch_upload_from_csv.py` - Specialized CSV batch processor
- `convert_cloudinary_urls.py` - Format conversion for existing Cloudinary images

---

**Author**: Tsitohaina  
**Updated**: November 2024  
**Version**: Unified Enterprise Edition
