import os
import json
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

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
            'cluster': 'cluster0.lwbmkvo.mongodb.net'
        }
    except Exception as e:
        print(f"Error loading MongoDB credentials: {e}")
        raise

def get_mongodb_client():
    """Create and return a MongoDB client"""
    try:
        creds = load_mongodb_credentials()
        connection_string = (
            f"mongodb+srv://{creds['username']}:{creds['password']}"
            f"@{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        )
        return MongoClient(connection_string, serverSelectionTimeoutMS=10000)
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None

def export_collection_to_csv(collection, output_dir, db_name, collection_name):
    """Export a collection to CSV"""
    try:
        # Create output directory if it doesn't exist
        db_dir = os.path.join(output_dir, db_name)
        os.makedirs(db_dir, exist_ok=True)
        
        # Create output file path
        output_file = os.path.join(db_dir, f"{collection_name}.csv")
        
        # Get all documents
        cursor = collection.find({})
        
        # Convert to list of dicts and handle ObjectId serialization
        documents = []
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            if '_id' in doc and isinstance(doc['_id'], ObjectId):
                doc['_id'] = str(doc['_id'])
            documents.append(doc)
        
        if not documents:
            print(f"  - {collection_name}: No documents found")
            return False
            
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(documents)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"  - {collection_name}: Successfully exported {len(documents)} documents to {output_file}")
        return True
        
    except Exception as e:
        print(f"  - {collection_name}: Error exporting - {str(e)}")
        return False

def main():
    print("=== MongoDB Export All to CSV ===")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    client = get_mongodb_client()
    if not client:
        return
    
    try:
        # Get list of all databases (except admin, local, and config)
        db_names = [db_name for db_name in client.list_database_names() 
                   if db_name not in ['admin', 'local', 'config']]
        
        total_collections = 0
        successful_exports = 0
        
        for db_name in db_names:
            print(f"\nExporting database: {db_name}")
            db = client[db_name]
            
            # Get all collections in the database
            collections = db.list_collection_names()
            total_collections += len(collections)
            
            for collection_name in collections:
                if collection_name.startswith('system.'):
                    continue  # Skip system collections
                    
                collection = db[collection_name]
                if export_collection_to_csv(collection, output_dir, db_name, collection_name):
                    successful_exports += 1
        
        print(f"\n=== Export Summary ===")
        print(f"Total collections found: {total_collections}")
        print(f"Successfully exported: {successful_exports}")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
    finally:
        client.close()
        print("\nExport completed.")

if __name__ == "__main__":
    main()
