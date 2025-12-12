import sys
import os
from pymongo import MongoClient

def load_mongodb_credentials():
    """Load MongoDB credentials from environment variables or file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to read from environment variables first
        password = os.getenv('MONGODB_PASSWORD')
        
        # If not in environment, try to read from file in the same directory
        password_file = os.path.join(script_dir, 'mongo_password.txt')
        if not password and os.path.exists(password_file):
            with open(password_file, 'r', encoding='utf-8') as f:
                password = f.read().strip()
                
        if not password:
            raise ValueError(f"MongoDB password not found. Please set MONGODB_PASSWORD environment variable or create {password_file}")
            
        return {
            'username': 'x25113186_db_user',
            'password': password,
            'cluster': 'cluster0.lwbmkvo.mongodb.net',
            'database': 'my_database'
        }
    except Exception as e:
        print(f"Error loading MongoDB credentials: {e}")
        raise


def print_safe(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main():
    print_safe("=== MongoDB Database Lister ===")
    
    try:
        # Load credentials
        creds = load_mongodb_credentials()
        
        # Create connection string
        connection_string = (
            f"mongodb+srv://{creds['username']}:{creds['password']}"
            f"@{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        )

        print_safe("Connecting to MongoDB...")
        print_safe(f"Connection string: mongodb+srv://{creds['username']}:{'*' * len(creds['password'])}@{creds['cluster']}/...")

        # Connect with timeout
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        print_safe("✓ Connected successfully!")

        # List all databases and collections
        print_safe("\n=== Databases ===")
        for db_name in client.list_database_names():
            print_safe(f"- {db_name}")
            
            # List collections in each database
            db = client[db_name]
            try:
                collections = db.list_collection_names()
                for col in collections:
                    print_safe(f"  - {col}")
            except Exception as e:
                print_safe(f"  (error listing collections: {e})")

    except Exception as e:
        print_safe("\n❌ Connection failed:")
        print_safe(str(e))
        print_safe("\nTroubleshooting:")
        print_safe("1. Check your internet connection")
        print_safe("2. Verify your IP is whitelisted in MongoDB Atlas")
        print_safe("3. Check your credentials in mongo_password.txt")
    finally:
        if 'client' in locals():
            client.close()

    if sys.platform == "win32":
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
