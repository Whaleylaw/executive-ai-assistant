from eaia.gmail import get_credentials
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_gmail():
    try:
        get_credentials()
        print("Gmail credentials setup successfully!")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure your client_secret file is in the eaia/.secrets directory")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(setup_gmail())
