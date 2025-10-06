# Cloudinary Bulk Image Uploader

This Python script allows you to bulk upload images to Cloudinary using multi-threading for improved performance.

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

1. Place your images in a folder (e.g., `images/`)

2. Run the script using:

   ```bash
   python main.py <folder_path> [max_workers]
   ```

   Arguments:

   - `folder_path`: Path to the folder containing images (required)
   - `max_workers`: Number of concurrent upload threads (optional, default: 10)

   Examples:

   ```bash
   python main.py ./images
   python main.py ./my_photos 15
   ```

## Features

- Multi-threaded uploads for better performance
- Progress tracking with success/failure counts
- Error logging for failed uploads
- CSV output with upload results
- Supports various image formats (.jpg, .jpeg, .png, .gif, .bmp, .webp, .svg)
- Preserves original file extensions
- Configuration through environment variables

## Output

The script will:

1. Create a CSV file with the same name as your input folder
2. The CSV will contain two columns:
   - `local_filename`: Original filename without extension
   - `cloudinary_url`: The Cloudinary URL for successful uploads or 'UPLOAD_FAILED' for failures

## Error Handling

- The script logs the first 10 errors encountered during upload
- Failed uploads are tracked and reported in the final summary
- Connection and configuration issues are detected before starting uploads
