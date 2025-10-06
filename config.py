import os
import cloudinary
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cloudinary Configuration
# Get your credentials from https://cloudinary.com/console
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Optional: Additional configuration settings
CLOUDINARY_FOLDER = "Tests"  # Folder name in Cloudinary
USE_FILENAME = True  # Use original filename
UNIQUE_FILENAME = True  # Add unique suffix to avoid duplicates