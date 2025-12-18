import os
import sys
import csv
import pymongo
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# Set console encoding to UTF-8 for Windows
if sys.platform.startswith('win'):
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def get_mongodb_connection():
    """Get MongoDB connection using existing credentials"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'paths.yaml')
        
        # Try to load from environment variables first
        creds = {
            'username': os.getenv('MONGODB_USERNAME'),
            'password': os.getenv('MONGODB_PASSWORD'),
            'cluster': os.getenv('MONGODB_CLUSTER'),
            'database': os.getenv('MONGODB_DATABASE', 'development_analysis')
        }
        
        # If any environment variables are missing, try to load from config file
        if None in creds.values():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                creds = {
                    'username': config.get('mongodb', {}).get('username'),
                    'password': config.get('mongodb', {}).get('password'),
                    'cluster': config.get('mongodb', {}).get('cluster'),
                    'database': config.get('mongodb', {}).get('database', 'development_analysis')
                }
        
        # Create connection string
        connection_string = (
            f"mongodb+srv://{creds['username']}:{creds['password']}"
            f"@{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        )
        
        # Connect to MongoDB
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        
        # Test the connection
        client.admin.command('ping')
        print("[SUCCESS] Successfully connected to MongoDB Atlas")
        
        return client, creds['database']
        
    except Exception as e:
        print(f"[ERROR] Error connecting to MongoDB: {e}")
        raise

def export_collection_to_csv(collection, output_dir):
    """Export a MongoDB collection to a CSV file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename based on collection name
    filename = f"{collection.name}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Get all documents from the collection
    cursor = collection.find()
    
    # Get field names (all unique keys from all documents)
    fieldnames = set()
    for doc in cursor.clone():
        fieldnames.update(doc.keys())
    
    # Convert ObjectId and datetime to string for CSV
    def clean_value(value):
        if isinstance(value, (ObjectId, datetime)):
            return str(value)
        return value
    
    # Write to CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
        writer.writeheader()
        
        try:
            for doc in cursor:
                # Clean the document values
                clean_doc = {k: clean_value(v) for k, v in doc.items()}
                writer.writerow(clean_doc)
            print(f"[SUCCESS] Exported {collection.name} to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to export {collection.name} to {filepath}: {e}")

def main():
    try:
        # Create output directory with timestamp
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_data')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, 'exports', f'export_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Connect to MongoDB
        client, db_name = get_mongodb_connection()
        db = client[db_name]
        
        # Get all collections
        collections = db.list_collection_names()
        print(f"\nFound {len(collections)} collections in database '{db_name}':")
        for i, col_name in enumerate(collections, 1):
            print(f"{i}. {col_name}")
        
        # Export each collection
        for col_name in collections:
            collection = db[col_name]
            export_collection_to_csv(collection, output_dir)
        
        print(f"\n[SUCCESS] All collections exported to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()
