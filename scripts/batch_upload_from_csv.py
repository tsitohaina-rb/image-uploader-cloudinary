#!/usr/bin/env python3
"""
Batch Upload from CSV - Process Multiple Google Drive Links

This script reads a CSV file containing Google Drive folder links and processes them
one by one using the Google Drive to Cloudinary upload functionality.

Features:
- Reads CSV files with Google Drive links
- Processes each folder link individually
- Configurable folder naming (from CSV column or auto-generated)
- Comprehensive logging and progress tracking
- Resume capability (skips already processed folders)
- Error handling and detailed reporting

Input CSV Format:
- Column with Google Drive links (default: 'links' column)
- Optional: Product Code column for custom folder naming
- Optional: Description column for reference

Usage:
    python scripts/batch_upload_from_csv.py input.csv
    python scripts/batch_upload_from_csv.py input.csv --link-column "links" --name-column "Product Code"
    python scripts/batch_upload_from_csv.py input.csv --max-workers 3 --recursive
"""

import os
import sys
import csv
import json
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

# Optional pandas import with fallback to csv module
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Google Drive upload functionality
from googledrive_tocloudinary import (
    upload_gdrive_folder_to_cloudinary,
    extract_folder_id_from_url,
    validate_folder_id,
    authenticate_google_drive,
    test_cloudinary_connection,
    test_google_drive_connection
)

# Setup directories
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

class BatchUploadProcessor:
    """Handles batch processing of CSV files with Google Drive links"""
    
    def __init__(self, csv_file, link_column='links', name_column=None, max_workers=5, recursive=True):
        self.csv_file = csv_file
        self.link_column = link_column
        self.name_column = name_column
        self.max_workers = max_workers
        self.recursive = recursive
        
        # Generate batch ID and setup logging
        self.batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.batch_name = Path(csv_file).stem
        self.setup_logging()
        
        # Initialize tracking
        self.processed_folders = []
        self.successful_uploads = 0
        self.failed_uploads = 0
        self.skipped_folders = 0
        
        # Load or create batch cache - use same naming pattern as log file
        self.batch_cache_file = os.path.join(CACHE_DIR, f'batch_cache_{self.batch_name}_{self.batch_id}.json')
        self.batch_cache = self.load_batch_cache()
    
    def setup_logging(self):
        """Setup logging for batch processing"""
        self.log_file = os.path.join(LOG_DIR, f'batch_upload_{self.batch_name}_{self.batch_id}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ],
            force=True  # Override any existing configuration
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch upload session started: {self.batch_id}")
        self.logger.info(f"CSV file: {self.csv_file}")
        self.logger.info(f"Log file: {self.log_file}")
    
    def load_batch_cache(self):
        """Load or create batch processing cache"""
        if os.path.exists(self.batch_cache_file):
            try:
                with open(self.batch_cache_file, 'r') as f:
                    cache = json.load(f)
                processed_count = len(cache.get('processed_folders', []))
                if processed_count > 0:
                    self.logger.info(f"🔄 RESUMING: Found existing batch cache with {processed_count} folders already processed")
                    self.logger.info(f"   Last run: {cache.get('last_updated', 'Unknown')}")
                return cache
            except Exception as e:
                self.logger.warning(f"Could not load batch cache: {e}")
        
        # Create new cache
        self.logger.info(f"🆕 STARTING: Creating new batch cache")
        return {
            'batch_id': self.batch_id,
            'csv_file': self.csv_file,
            'started_at': datetime.now().isoformat(),
            'processed_folders': [],
            'failed_folders': [],
            'summary': {}
        }
    
    def save_batch_cache(self):
        """Save batch processing cache"""
        try:
            self.batch_cache['last_updated'] = datetime.now().isoformat()
            with open(self.batch_cache_file, 'w') as f:
                json.dump(self.batch_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save batch cache: {e}")
    
    def generate_incremental_csv(self, folder_result):
        """Generate/update CSV file after each successful folder upload"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'batch_results_{self.batch_name}_latest.csv'
            csv_file_path = os.path.join(OUTPUT_DIR, csv_filename)
            
            # Collect all successful folders from cache
            all_successful = self.batch_cache.get('processed_folders', [])
            
            if not all_successful:
                return
            
            # Create CSV with comprehensive results
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'row_index', 'folder_name', 'drive_folder_name', 'folder_id', 
                    'total_images', 'successful_uploads', 'failed_uploads', 'skipped_uploads',
                    'processing_time_seconds', 'processed_at', 'drive_link', 'status'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for folder in all_successful:
                    writer.writerow({
                        'row_index': folder.get('index', ''),
                        'folder_name': folder.get('folder_name', ''),
                        'drive_folder_name': folder.get('drive_folder_name', ''),
                        'folder_id': folder.get('folder_id', ''),
                        'total_images': folder.get('total_images', 0),
                        'successful_uploads': folder.get('successful_uploads', 0),
                        'failed_uploads': folder.get('failed_uploads', 0),
                        'skipped_uploads': folder.get('skipped_uploads', 0),
                        'processing_time_seconds': folder.get('processing_time', 0),
                        'processed_at': folder.get('processed_at', ''),
                        'drive_link': folder.get('drive_link', ''),
                        'status': folder.get('status', '')
                    })
            
            self.logger.info(f"📊 CSV updated: {csv_file_path} ({len(all_successful)} folders)")
            
        except Exception as e:
            self.logger.warning(f"Could not generate incremental CSV: {e}")
    
    def create_individual_folder_backup(self, folder_result):
        """Create individual backup cache for each processed folder"""
        try:
            folder_id = folder_result.get('folder_id', 'unknown')
            backup_filename = f'folder_backup_{folder_id}_{self.batch_id}.json'
            backup_file_path = os.path.join(CACHE_DIR, backup_filename)
            
            # Save detailed folder result
            with open(backup_file_path, 'w') as f:
                json.dump(folder_result, f, indent=2)
            
            self.logger.debug(f"💾 Individual backup created: {backup_filename}")
            
        except Exception as e:
            self.logger.warning(f"Could not create individual folder backup: {e}")
    
    def read_csv_data(self):
        """Read and validate CSV data"""
        try:
            if PANDAS_AVAILABLE:
                # Use pandas for better handling
                df = pd.read_csv(self.csv_file)
                
                # Check if link column exists
                if self.link_column not in df.columns:
                    available_columns = list(df.columns)
                    self.logger.error(f"Link column '{self.link_column}' not found in CSV")
                    self.logger.error(f"Available columns: {available_columns}")
                    raise ValueError(f"Column '{self.link_column}' not found")
                
                # Check if name column exists (if specified)
                if self.name_column and self.name_column not in df.columns:
                    self.logger.warning(f"Name column '{self.name_column}' not found, will use row index")
                    self.name_column = None
                
                # Filter out rows with empty links
                df_filtered = df[df[self.link_column].notna() & (df[self.link_column] != '')]
                
                self.logger.info(f"CSV loaded successfully (using pandas):")
                self.logger.info(f"  Total rows: {len(df)}")
                self.logger.info(f"  Rows with valid links: {len(df_filtered)}")
                self.logger.info(f"  Link column: '{self.link_column}'")
                if self.name_column:
                    self.logger.info(f"  Name column: '{self.name_column}'")
                
                return df_filtered
            
            else:
                # Fallback to standard csv module
                data = []
                with open(self.csv_file, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    # Check if link column exists
                    if not reader.fieldnames or self.link_column not in reader.fieldnames:
                        available_columns = list(reader.fieldnames) if reader.fieldnames else []
                        self.logger.error(f"Link column '{self.link_column}' not found in CSV")
                        self.logger.error(f"Available columns: {available_columns}")
                        raise ValueError(f"Column '{self.link_column}' not found")
                    
                    # Check if name column exists (if specified)
                    if self.name_column and self.name_column not in reader.fieldnames:
                        self.logger.warning(f"Name column '{self.name_column}' not found, will use row index")
                        self.name_column = None
                    
                    # Read all rows and filter empty links
                    for row in reader:
                        if row.get(self.link_column, '').strip():
                            data.append(row)
                
                # Create a simple dataframe-like structure
                class SimpleDataFrame:
                    def __init__(self, data):
                        self.data = data
                    
                    def __len__(self):
                        return len(self.data)
                    
                    def __bool__(self):
                        return len(self.data) > 0
                    
                    @property
                    def empty(self):
                        return len(self.data) == 0
                    
                    def iterrows(self):
                        for i, row in enumerate(self.data):
                            yield i, row
                
                df_filtered = SimpleDataFrame(data)
                
                self.logger.info(f"CSV loaded successfully (using standard csv):")
                self.logger.info(f"  Rows with valid links: {len(df_filtered)}")
                self.logger.info(f"  Link column: '{self.link_column}'")
                if self.name_column:
                    self.logger.info(f"  Name column: '{self.name_column}'")
                
                return df_filtered
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            raise
    
    def extract_folder_info(self, row, index):
        """Extract folder information from CSV row"""
        drive_link = str(row[self.link_column]).strip()
        
        # Extract folder ID
        folder_id = extract_folder_id_from_url(drive_link)
        if not folder_id:
            self.logger.error(f"Row {index + 1}: Could not extract folder ID from: {drive_link}")
            return None
        
        # Determine folder name for Cloudinary
        if self.name_column and str(row.get(self.name_column, '')).strip():
            folder_name = str(row[self.name_column]).strip()
            # Clean folder name for filesystem safety
            folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('|', '_').replace('?', '_').replace('*', '_').strip()
        else:
            folder_name = f"folder_{index + 1}_{folder_id[:8]}"
        
        # Get description if available
        description = ""
        if 'description' in row and row['description'] and str(row['description']).strip():
            description = str(row['description']).strip()[:100]  # Truncate for logging
        
        return {
            'index': index + 1,
            'folder_id': folder_id,
            'folder_name': folder_name,
            'drive_link': drive_link,
            'description': description,
            'row_data': dict(row)
        }
    
    def is_folder_processed(self, folder_id):
        """Check if folder was already processed successfully"""
        return any(
            processed['folder_id'] == folder_id and processed.get('status') == 'success'
            for processed in self.batch_cache.get('processed_folders', [])
        )
    
    def process_single_folder(self, folder_info):
        """Process a single Google Drive folder"""
        folder_id = folder_info['folder_id']
        folder_name = folder_info['folder_name']
        index = folder_info['index']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing Row {index}: {folder_name}")
        self.logger.info(f"Folder ID: {folder_id}")
        self.logger.info(f"Drive Link: {folder_info['drive_link']}")
        if folder_info['description']:
            self.logger.info(f"Description: {folder_info['description']}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Validate folder access
            service = authenticate_google_drive()
            if not service:
                raise Exception("Failed to authenticate Google Drive")
            
            is_valid, drive_folder_name, error_msg = validate_folder_id(service, folder_id)
            if not is_valid:
                raise Exception(f"Folder validation failed: {error_msg}")
            
            self.logger.info(f"✅ Folder access confirmed: '{drive_folder_name}'")
            
            # Use the actual Drive folder name if no custom name provided
            if self.name_column is None:
                folder_name = drive_folder_name
            
            # Perform the upload
            self.logger.info(f"🚀 Starting upload to Cloudinary folder: '{folder_name}'")
            results = upload_gdrive_folder_to_cloudinary(
                folder_id=folder_id,
                folder_name=folder_name,
                max_workers=self.max_workers,
                recursive=self.recursive
            )
            
            elapsed_time = time.time() - start_time
            
            # Analyze results
            if results:
                success_count = sum(1 for r in results if r.get('status') == 'success')
                failed_count = sum(1 for r in results if r.get('status') == 'failed')
                skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
                
                self.logger.info(f"✅ Folder processing completed:")
                self.logger.info(f"  Processing time: {elapsed_time:.2f} seconds")
                self.logger.info(f"  Total images: {len(results)}")
                self.logger.info(f"  Successful uploads: {success_count}")
                self.logger.info(f"  Failed uploads: {failed_count}")
                self.logger.info(f"  Skipped (cached): {skipped_count}")
                
                # Update batch cache
                folder_result = {
                    'folder_id': folder_id,
                    'folder_name': folder_name,
                    'drive_folder_name': drive_folder_name,
                    'index': index,
                    'status': 'success',
                    'processing_time': elapsed_time,
                    'total_images': len(results),
                    'successful_uploads': success_count,
                    'failed_uploads': failed_count,
                    'skipped_uploads': skipped_count,
                    'processed_at': datetime.now().isoformat(),
                    'drive_link': folder_info['drive_link']
                }
                
                self.batch_cache['processed_folders'].append(folder_result)
                self.successful_uploads += 1
                
                # Create individual folder backup
                self.create_individual_folder_backup(folder_result)
                
                # Generate incremental CSV after each successful upload
                self.generate_incremental_csv(folder_result)
                
            else:
                self.logger.warning(f"⚠️ No results returned for folder: {folder_name}")
                folder_result = {
                    'folder_id': folder_id,
                    'folder_name': folder_name,
                    'index': index,
                    'status': 'no_results',
                    'processing_time': elapsed_time,
                    'processed_at': datetime.now().isoformat(),
                    'drive_link': folder_info['drive_link'],
                    'error': 'No results returned'
                }
                self.batch_cache['failed_folders'].append(folder_result)
                self.failed_uploads += 1
            
            self.save_batch_cache()
            return True
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"❌ Failed to process folder: {folder_name}")
            self.logger.error(f"Error: {error_msg}")
            
            # Update batch cache with failure
            folder_result = {
                'folder_id': folder_id,
                'folder_name': folder_name,
                'index': index,
                'status': 'failed',
                'processing_time': elapsed_time,
                'processed_at': datetime.now().isoformat(),
                'drive_link': folder_info['drive_link'],
                'error': error_msg
            }
            self.batch_cache['failed_folders'].append(folder_result)
            self.failed_uploads += 1
            self.save_batch_cache()
            
            return False
    
    def generate_final_report(self):
        """Generate final batch processing report"""
        total_processed = self.successful_uploads + self.failed_uploads + self.skipped_folders
        
        # Update batch cache summary
        self.batch_cache['summary'] = {
            'completed_at': datetime.now().isoformat(),
            'total_folders_processed': total_processed,
            'successful_uploads': self.successful_uploads,
            'failed_uploads': self.failed_uploads,
            'skipped_folders': self.skipped_folders,
            'success_rate': f"{(self.successful_uploads / max(total_processed, 1)) * 100:.1f}%"
        }
        self.save_batch_cache()
        
        # Create summary report file
        report_file = os.path.join(OUTPUT_DIR, f'batch_upload_report_{self.batch_name}_{self.batch_id}.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"Batch Upload Report\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Session ID: {self.batch_id}\n")
            f.write(f"CSV File: {self.csv_file}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total folders processed: {total_processed}\n")
            f.write(f"  Successful uploads: {self.successful_uploads}\n")
            f.write(f"  Failed uploads: {self.failed_uploads}\n")
            f.write(f"  Skipped folders: {self.skipped_folders}\n")
            f.write(f"  Success rate: {self.batch_cache['summary']['success_rate']}\n\n")
            
            # Successful folders
            if self.batch_cache.get('processed_folders'):
                f.write(f"Successful Uploads:\n")
                f.write(f"{'-'*30}\n")
                for folder in self.batch_cache['processed_folders']:
                    f.write(f"Row {folder['index']}: {folder['folder_name']}\n")
                    f.write(f"  Images: {folder.get('total_images', 0)} "
                           f"(Success: {folder.get('successful_uploads', 0)}, "
                           f"Failed: {folder.get('failed_uploads', 0)}, "
                           f"Skipped: {folder.get('skipped_uploads', 0)})\n")
                    f.write(f"  Time: {folder.get('processing_time', 0):.1f}s\n\n")
            
            # Failed folders
            if self.batch_cache.get('failed_folders'):
                f.write(f"\nFailed Uploads:\n")
                f.write(f"{'-'*20}\n")
                for folder in self.batch_cache['failed_folders']:
                    f.write(f"Row {folder['index']}: {folder['folder_name']}\n")
                    f.write(f"  Error: {folder.get('error', 'Unknown error')}\n\n")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH PROCESSING COMPLETED")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total folders processed: {total_processed}")
        self.logger.info(f"Successful uploads: {self.successful_uploads}")
        self.logger.info(f"Failed uploads: {self.failed_uploads}")
        self.logger.info(f"Skipped folders: {self.skipped_folders}")
        self.logger.info(f"Success rate: {self.batch_cache['summary']['success_rate']}")
        self.logger.info(f"")
        self.logger.info(f"📄 Detailed report saved: {report_file}")
        self.logger.info(f"📄 Detailed logs: {self.log_file}")
        self.logger.info(f"💾 Batch cache: {self.batch_cache_file}")
        self.logger.info(f"📊 Live results CSV: data/output/batch_results_{self.batch_name}_latest.csv")
        self.logger.info(f"🔄 Individual backups: data/cache/folder_backup_*.json")
        self.logger.info(f"")
        self.logger.info(f"💡 RECOVERY TIP: If interrupted, rerun the same command to resume automatically")
    
    def run(self):
        """Run the batch upload process"""
        try:
            # Test connections first
            self.logger.info("Testing connections...")
            
            # Test Cloudinary
            is_configured, message = test_cloudinary_connection()
            if not is_configured:
                self.logger.error(f"Cloudinary configuration issue: {message}")
                return False
            self.logger.info("✅ Cloudinary connection verified")
            
            # Test Google Drive
            is_connected, message = test_google_drive_connection()
            if not is_connected:
                self.logger.error(f"Google Drive connection issue: {message}")
                return False
            self.logger.info("✅ Google Drive connection verified")
            
            # Read CSV data
            self.logger.info("\nReading CSV data...")
            df = self.read_csv_data()
            
            if df.empty:
                self.logger.error("No valid rows found in CSV file")
                return False
            
            # Process each folder
            self.logger.info(f"\nStarting batch processing of {len(df)} folders...")
            self.logger.info(f"Max workers per folder: {self.max_workers}")
            self.logger.info(f"Recursive scanning: {self.recursive}")
            self.logger.info(f"💾 Progress saved to: {self.batch_cache_file}")
            self.logger.info(f"📊 Live CSV updates: data/output/batch_results_{self.batch_name}_latest.csv")
            self.logger.info(f"🔄 Crash recovery: Script can be safely restarted to resume from last completed folder")
            
            total_to_process = 0
            for index, row in df.iterrows():
                folder_info = self.extract_folder_info(row, index)
                if folder_info and not self.is_folder_processed(folder_info['folder_id']):
                    total_to_process += 1
            
            if total_to_process == 0:
                self.logger.info("🎉 All folders already processed! Nothing to do.")
                self.generate_final_report()
                return True
            
            self.logger.info(f"📝 Folders to process: {total_to_process} (remaining)")
            
            processed_in_session = 0
            
            processed_in_session = 0
            
            for index, row in df.iterrows():
                # Extract folder information
                folder_info = self.extract_folder_info(row, index)
                if not folder_info:
                    self.failed_uploads += 1
                    continue
                
                # Check if already processed
                if self.is_folder_processed(folder_info['folder_id']):
                    self.logger.info(f"⏭️ Skipping Row {folder_info['index']}: {folder_info['folder_name']} (already processed)")
                    self.skipped_folders += 1
                    continue
                
                # Process the folder
                self.logger.info(f"\n🚀 Processing {processed_in_session + 1}/{total_to_process} folders in this session...")
                success = self.process_single_folder(folder_info)
                processed_in_session += 1
                
                if success:
                    self.logger.info(f"✅ Session progress: {processed_in_session}/{total_to_process} completed")
                else:
                    self.logger.info(f"❌ Session progress: {processed_in_session}/{total_to_process} attempted")
                
                # Add a small delay between folders to avoid rate limiting
                if index < len(df) - 1:  # Don't sleep after the last folder
                    time.sleep(2)
            
            # Generate final report
            self.generate_final_report()
            return True
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return False

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch upload from CSV with Google Drive links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python scripts/batch_upload_from_csv.py data/input/products.csv
  
  # Specify custom columns
  python scripts/batch_upload_from_csv.py data/input/products.csv --link-column "links" --name-column "Product Code"
  
  # Adjust performance settings
  python scripts/batch_upload_from_csv.py data/input/products.csv --max-workers 3 --no-recursive
  
  # Resume previous batch (script automatically detects and skips processed folders)
  python scripts/batch_upload_from_csv.py data/input/products.csv
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to CSV file containing Google Drive links'
    )
    
    parser.add_argument(
        '--link-column',
        default='links',
        help='Name of the column containing Google Drive links (default: "links")'
    )
    
    parser.add_argument(
        '--name-column',
        help='Name of the column to use for Cloudinary folder names (default: auto-generate)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of concurrent upload threads per folder (default: 5)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Disable recursive scanning of subfolders'
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Create processor and run
    processor = BatchUploadProcessor(
        csv_file=args.csv_file,
        link_column=args.link_column,
        name_column=args.name_column,
        max_workers=args.max_workers,
        recursive=not args.no_recursive
    )
    
    success = processor.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())