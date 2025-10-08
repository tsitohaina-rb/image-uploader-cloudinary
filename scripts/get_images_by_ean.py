"""
Get Product Images by EAN
=========================
Extract all product images URLs by EAN codes and export to CSV

Usage: python get_images_by_ean.py

Requirements:
- mysql-connector-python
- python-dotenv

Install with: pip install mysql-connector-python python-dotenv
"""

try:
    import mysql.connector
except ImportError:
    print("ERROR: mysql-connector-python not installed")
    print("Install with: pip install mysql-connector-python")
    exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv not installed")
    print("Install with: pip install python-dotenv")
    exit(1)

import os
import csv
import sys
from datetime import datetime

load_dotenv()

class ImageExtractor:
    """Extract product images by EAN"""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME'),
                port=int(os.getenv('DB_PORT', 3306))
            )
            
            if self.connection.is_connected():
                print(f"Connected to database: {os.getenv('DB_NAME')}")
                return True
        except mysql.connector.Error as e:
            print(f"Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")
    
    def get_images_by_ean(self, ean_codes, debug=False):
        """
        Get all product images for given EAN codes
        Returns dict with EAN as key and list of image URLs as value
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if isinstance(ean_codes, str):
                ean_codes = [ean_codes]
            
            clean_eans = [str(ean).strip() for ean in ean_codes if str(ean).strip()]
            
            if not clean_eans:
                print("No valid EAN codes provided")
                return {}
            
            print(f"\nSearching images for {len(clean_eans)} EAN code(s)...")
            
            # Debug: First show the product_group_id for the EAN
            if debug:
                debug_query = "SELECT ean, idproduit_group FROM produits_view3 WHERE ean IN ({})".format(','.join(['%s'] * len(clean_eans)))
                cursor.execute(debug_query, clean_eans)
                debug_results = cursor.fetchall()
                print("\n[DEBUG] Products found:")
                for dr in debug_results:
                    print(f"  EAN: {dr['ean']} | Product Group ID: {dr['idproduit_group']}")
                
                # Show gallery entries for these product groups
                if debug_results:
                    group_ids = [dr['idproduit_group'] for dr in debug_results]
                    gallery_query = """
                    SELECT idimage, idproduit_group, position, ext, status 
                    FROM produits_gallery 
                    WHERE idproduit_group IN ({}) 
                    ORDER BY idproduit_group, position
                    """.format(','.join(['%s'] * len(group_ids)))
                    cursor.execute(gallery_query, group_ids)
                    gallery_results = cursor.fetchall()
                    print("\n[DEBUG] Gallery entries:")
                    for gr in gallery_results:
                        print(f"  Group: {gr['idproduit_group']} | Pos: {gr['position']} | "
                              f"Image: {gr['idimage']}.{gr['ext']} | Status: {gr['status']}")
            
            # Query to get products with their images
            # Note: position in produits_gallery starts at 0
            # position 0 = image_1, position 1 = image_2, etc.
            query = """
            SELECT 
                p.ean,
                p.idproduit_group,
                CASE 
                    WHEN g1.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g1.idimage, '.', g1.ext)
                    ELSE ''
                END as image_1,
                CASE 
                    WHEN g2.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g2.idimage, '.', g2.ext)
                    ELSE ''
                END as image_2,
                CASE 
                    WHEN g3.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g3.idimage, '.', g3.ext)
                    ELSE ''
                END as image_3,
                CASE 
                    WHEN g4.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g4.idimage, '.', g4.ext)
                    ELSE ''
                END as image_4,
                CASE 
                    WHEN g5.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g5.idimage, '.', g5.ext)
                    ELSE ''
                END as image_5,
                CASE 
                    WHEN g6.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g6.idimage, '.', g6.ext)
                    ELSE ''
                END as image_6,
                CASE 
                    WHEN g7.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g7.idimage, '.', g7.ext)
                    ELSE ''
                END as image_7,
                CASE 
                    WHEN g8.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g8.idimage, '.', g8.ext)
                    ELSE ''
                END as image_8,
                CASE 
                    WHEN g9.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g9.idimage, '.', g9.ext)
                    ELSE ''
                END as image_9,
                CASE 
                    WHEN g10.idimage IS NOT NULL THEN CONCAT('https://cdn.bazarchic.com/i/tmp/', g10.idimage, '.', g10.ext)
                    ELSE ''
                END as image_10
            FROM produits_view3 p
            LEFT JOIN produits_gallery g1 ON p.idproduit_group = g1.idproduit_group AND g1.position = 0 AND g1.status = 'on'
            LEFT JOIN produits_gallery g2 ON p.idproduit_group = g2.idproduit_group AND g2.position = 1 AND g2.status = 'on'
            LEFT JOIN produits_gallery g3 ON p.idproduit_group = g3.idproduit_group AND g3.position = 2 AND g3.status = 'on'
            LEFT JOIN produits_gallery g4 ON p.idproduit_group = g4.idproduit_group AND g4.position = 3 AND g4.status = 'on'
            LEFT JOIN produits_gallery g5 ON p.idproduit_group = g5.idproduit_group AND g5.position = 4 AND g5.status = 'on'
            LEFT JOIN produits_gallery g6 ON p.idproduit_group = g6.idproduit_group AND g6.position = 5 AND g6.status = 'on'
            LEFT JOIN produits_gallery g7 ON p.idproduit_group = g7.idproduit_group AND g7.position = 6 AND g7.status = 'on'
            LEFT JOIN produits_gallery g8 ON p.idproduit_group = g8.idproduit_group AND g8.position = 7 AND g8.status = 'on'
            LEFT JOIN produits_gallery g9 ON p.idproduit_group = g9.idproduit_group AND g9.position = 8 AND g9.status = 'on'
            LEFT JOIN produits_gallery g10 ON p.idproduit_group = g10.idproduit_group AND g10.position = 9 AND g10.status = 'on'
            WHERE p.ean IN ({})
            AND p.status = 'on'
            """.format(','.join(['%s'] * len(clean_eans)))
            
            cursor.execute(query, clean_eans)
            results = cursor.fetchall()
            
            cursor.close()
            
            # Group by EAN (in case multiple products have same EAN)
            images_by_ean = {}
            for row in results:
                ean = row['ean']
                if ean not in images_by_ean:
                    images_by_ean[ean] = row
            
            print(f"Found images for {len(images_by_ean)} EAN code(s)")
            
            if debug:
                print("\n[DEBUG] Final results:")
                for ean, data in images_by_ean.items():
                    print(f"\n  EAN: {ean}")
                    for i in range(1, 11):
                        img_url = data.get(f'image_{i}', '')
                        if img_url:
                            print(f"    image_{i}: {img_url}")
            
            return images_by_ean
            
        except mysql.connector.Error as e:
            print(f"Database error: {e}")
            return {}
    
    def export_to_csv(self, ean_codes, filename=None):
        """
        Export product images by EAN to CSV
        Creates 3 separate files:
        - ALL EANs (found + not found combined)
        - Found EANs only
        - Not found EANs only
        """
        try:
            if isinstance(ean_codes, str):
                ean_codes = [ean_codes]
            
            clean_eans = [str(ean).strip() for ean in ean_codes if str(ean).strip()]
            
            if not clean_eans:
                print("No valid EAN codes provided")
                return None, None, None
            
            # Get images data
            images_data = self.get_images_by_ean(clean_eans)
            
            # Ensure output directory exists
            output_dir = os.path.join('data', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if filename:
                # Get the base name from the input file (without extension and path)
                base_name = os.path.splitext(os.path.basename(filename))[0]
            else:
                base_name = 'images_by_ean'
                
            # Create filenames with input filename and timestamp
            filename_all = os.path.join(output_dir, f"{base_name}_ALL_{timestamp}.csv")
            filename_found = os.path.join(output_dir, f"{base_name}_FOUND_{timestamp}.csv")
            filename_not_found = os.path.join(output_dir, f"{base_name}_NOT_FOUND_{timestamp}.csv")
            
            headers = ['ean', 'image_1', 'image_2', 'image_3', 'image_4', 
                      'image_5', 'image_6', 'image_7', 'image_8', 'image_9', 'image_10']
            
            # Track statistics
            found_eans = []
            not_found_eans = []
            
            # Write ALL EANs file (found + not found combined)
            with open(filename_all, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for ean in clean_eans:
                    if ean in images_data:
                        found_eans.append(ean)
                        row_data = images_data[ean]
                        writer.writerow([
                            ean,
                            row_data.get('image_1', ''),
                            row_data.get('image_2', ''),
                            row_data.get('image_3', ''),
                            row_data.get('image_4', ''),
                            row_data.get('image_5', ''),
                            row_data.get('image_6', ''),
                            row_data.get('image_7', ''),
                            row_data.get('image_8', ''),
                            row_data.get('image_9', ''),
                            row_data.get('image_10', '')
                        ])
                    else:
                        not_found_eans.append(ean)
                        writer.writerow([ean, '', '', '', '', '', '', '', '', '', ''])
            
            # Write FOUND EANs file
            with open(filename_found, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for ean in found_eans:
                    row_data = images_data[ean]
                    writer.writerow([
                        ean,
                        row_data.get('image_1', ''),
                        row_data.get('image_2', ''),
                        row_data.get('image_3', ''),
                        row_data.get('image_4', ''),
                        row_data.get('image_5', ''),
                        row_data.get('image_6', ''),
                        row_data.get('image_7', ''),
                        row_data.get('image_8', ''),
                        row_data.get('image_9', ''),
                        row_data.get('image_10', '')
                    ])
            
            # Write NOT FOUND EANs file
            with open(filename_not_found, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for ean in not_found_eans:
                    writer.writerow([ean, '', '', '', '', '', '', '', '', '', ''])
            
            # Calculate file sizes
            file_size_all = os.path.getsize(filename_all) / 1024
            file_size_found = os.path.getsize(filename_found) / 1024
            file_size_not_found = os.path.getsize(filename_not_found) / 1024
            
            print(f"\nExport completed!")
            print("=" * 60)
            print(f"ALL EANs File: {filename_all}")
            print(f"  Total EANs: {len(clean_eans)}")
            print(f"  File size: {file_size_all:.2f} KB")
            print()
            print(f"FOUND EANs File: {filename_found}")
            print(f"  EANs found: {len(found_eans)}")
            print(f"  File size: {file_size_found:.2f} KB")
            print()
            print(f"NOT FOUND EANs File: {filename_not_found}")
            print(f"  EANs not found: {len(not_found_eans)}")
            print(f"  File size: {file_size_not_found:.2f} KB")
            print()
            print(f"Summary:")
            print(f"  Total EANs searched: {len(clean_eans)}")
            print(f"  Found: {len(found_eans)} ({len(found_eans)/len(clean_eans)*100:.1f}%)")
            print(f"  Not Found: {len(not_found_eans)} ({len(not_found_eans)/len(clean_eans)*100:.1f}%)")
            
            return filename_all, filename_found, filename_not_found
            
        except Exception as e:
            print(f"Export error: {e}")
            return None, None, None


def main():
    """Main application"""
    print("Get Product Images by EAN")
    print("=" * 50)
    
    extractor = ImageExtractor()
    
    if not extractor.connection:
        print("Cannot proceed without database connection")
        return
    
    while True:
        print("\nOptions:")
        print("1. Search single EAN")
        print("2. Search multiple EANs (comma-separated)")
        print("3. Load EANs from file")
        print("4. Debug mode - Test single EAN with detailed output")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-4): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            
            elif choice == "1":
                ean = input("Enter EAN code: ").strip()
                if ean:
                    extractor.export_to_csv([ean])
                else:
                    print("No EAN code provided")
            
            elif choice == "2":
                eans_input = input("Enter EAN codes (comma-separated): ").strip()
                if eans_input:
                    eans = [ean.strip() for ean in eans_input.split(',')]
                    extractor.export_to_csv(eans)
                else:
                    print("No EAN codes provided")
            
            elif choice == "3":
                file_path = input("Enter path to text file with EAN codes: ").strip()
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            eans = [line.strip() for line in f if line.strip()]
                        
                        if eans:
                            print(f"Loaded {len(eans)} EAN codes from file")
                            extractor.export_to_csv(eans, filename=file_path)
                        else:
                            print("No valid EAN codes found in file")
                    except Exception as e:
                        print(f"Error reading file: {e}")
                else:
                    print("File not found")
            
            elif choice == "4":
                ean = input("Enter EAN code for debug: ").strip()
                if ean:
                    print("\n" + "="*60)
                    print("DEBUG MODE - Detailed Image Retrieval")
                    print("="*60)
                    extractor.get_images_by_ean([ean], debug=True)
                else:
                    print("No EAN code provided")
            
            else:
                print("Invalid option. Please select 0-4.")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    extractor.close()


if __name__ == "__main__":
    main()