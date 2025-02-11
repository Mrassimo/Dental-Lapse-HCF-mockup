"""
HCF Data Download Script
-----------------------

This script downloads required datasets for the HCF data science project.
It's designed to be beginner-friendly with detailed explanations.

Key Features:
- Downloads health insurance data
- Downloads member data
- Sets up directory structure
- Handles Kaggle API authentication

Note: You'll need a Kaggle account and API key to use this script.
See README.md for setup instructions.

Author: [Your Name]
Last Updated: [Date]
"""

# Essential Python Packages
# -----------------------
# os: Operating system interface (like a Python version of Terminal/Command Prompt)
import os

# kaggle: API for downloading datasets from Kaggle
import kaggle

# pathlib: Modern way to handle file paths (works on Windows & Mac)
from pathlib import Path


def setup_directories():
    """
    Create necessary folders if they don't exist.
    
    This is like creating folders in Finder/Explorer, but automated.
    The folders will store:
    - raw: Original downloaded data
    - processed: Cleaned and transformed data
    - synthetic: Generated test data
    """
    print("Setting up directory structure...")
    
    # List of folders we need
    base_dirs = ['raw', 'processed', 'synthetic']
    
    # Create each folder
    for dir_name in base_dirs:
        folder_path = Path(f'data/{dir_name}')
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_name} directory")


def download_datasets():
    """
    Download required datasets from Kaggle.
    
    We're using sample datasets that we'll modify to match
    HCF's context. The data includes:
    - Health insurance claims
    - Member information
    """
    print("\nDownloading datasets...")
    
    try:
        # Download health insurance dataset
        print("Downloading health insurance data...")
        kaggle.api.dataset_download_files(
            'teertha/ushealthinsurancedataset',
            path='data/raw/health_insurance',
            unzip=True
        )
        print("✓ Health insurance data downloaded")
        
        # Download member dataset
        print("\nDownloading member data...")
        kaggle.api.dataset_download_files(
            'usmanfarid/customer-churn-dataset-for-life-insurance-industry',
            path='data/raw/member_data',
            unzip=True
        )
        print("✓ Member data downloaded")
        
    except Exception as e:
        print(f"\nError downloading datasets: {e}")
        print("\nTo use the Kaggle API, please:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to 'Account' -> 'Create API Token'")
        print("3. Download kaggle.json and save it to:")
        print("   - Mac: ~/.kaggle/kaggle.json")
        print("   - Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
        print("4. Set correct permissions:")
        print("   - Mac: chmod 600 ~/.kaggle/kaggle.json")
        print("   - Windows: No additional steps needed")
        return False
    
    return True


def main():
    """
    Main function to run the download process.
    
    This coordinates the whole download process:
    1. Creates necessary folders
    2. Downloads the datasets
    3. Provides next steps
    """
    print("Starting HCF Data Science Project Setup...")
    print("-" * 50)
    
    # Create directory structure
    setup_directories()
    
    # Download datasets
    if download_datasets():
        print("\nSetup complete! Next steps:")
        print("1. Check data/raw/ for downloaded datasets")
        print("2. Run preprocessing.py to prepare the data")
        print("3. Start with 01_data_prep.ipynb")
    else:
        print("\nSetup incomplete. Please fix the issues above and try again.")


# Only run this script if it's being run directly
# (not being imported by another script)
if __name__ == "__main__":
    main() 