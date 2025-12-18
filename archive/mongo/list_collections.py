import pymongo
import json
import sys
import os
from pprint import pprint

def load_mongodb_credentials():
    """Load MongoDB credentials from environment variables or file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to read from environment variables first
        password = os.getenv('MONGODB_PASSWORD')
        
        # If not in environment, try to read from file in the same directory
        password_file = os.path.join(script_dir, 'mongo_password.countries_of_interest 1.txt')
        if not password and os.path.exists(password_file):
            with open(password_file, 'r', encoding='utf-8') as f:
                password = f.read().strip()
                
        if not password:
            raise ValueError(f"MongoDB password not found. Please set MONGODB_PASSWORD environment variable or create {password_file}")
            
        return {
            'username': 'x25113186_db_user',
            'password': password,
            'cluster': 'cluster0.lwbmkvo.mongodb.net',
            'database': 'development_analysis'
        }
    except Exception as e:
        print(f"Error loading MongoDB credentials: {e}")
        raise

def print_safe(msg):
    try:
        if sys.platform.startswith('win'):
            # On Windows, use the console's encoding
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        print(msg)
    except Exception as e:
        # If all else fails, try ASCII with replacement
        print(str(msg).encode('ascii', 'replace').decode('ascii'))

def get_collections():
    print("Connecting to MongoDB Atlas...")
    
    try:
        # Load credentials
        creds = load_mongodb_credentials()
        
        # Create connection string
        connection_string = (
            f"mongodb+srv://{creds['username']}:{creds['password']}@"
            f"{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        )
        
        # Connect to MongoDB Atlas
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        
        # Test the connection
        client.admin.command('ping')
        print("[SUCCESS] Connected to MongoDB Atlas")
        
        # Get the development_analysis database
        db = client[creds['database']]
        
        # Get list of collections
        collections = db.list_collection_names()
        
        print("\n=== Collections in development_analysis database ===")
        if not collections:
            print("No collections found in 'development_analysis' database.")
        else:
            for collection in collections:
                count = db[collection].count_documents({})
                print(f"- {collection} (documents: {count})")
            
        # Display document count for each collection
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"\nCollection: {collection_name}")
            print(f"Total documents: {count}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client.close()

if __name__ == "__main__":
    get_collections()
