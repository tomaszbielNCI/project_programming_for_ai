from pymongo import MongoClient
from pprint import pprint
from bson.objectid import ObjectId
import os

def load_mongodb_credentials():
    """Load MongoDB credentials from environment variables or file"""
    try:
        # Try to read from environment variables first
        password = os.getenv('MONGODB_PASSWORD')
        
        # If not in environment, try to read from file
        if not password and os.path.exists('mongo_atlas_password.txt'):
            with open('mongo_atlas_password.txt', 'r') as f:
                password = f.read().strip()
                
        if not password:
            raise ValueError("MongoDB password not found. Please set MONGODB_PASSWORD environment variable or create mongo_atlas_password.txt")
            
        return {
            'username': 'x25113186_db_user',
            'password': password,
            'cluster': 'cluster0.lwbmkvo.mongodb.net',
            'database': 'my_database'
        }
    except Exception as e:
        print(f"Error loading MongoDB credentials: {e}")
        raise

def connect_to_mongodb():
    try:
        # Load credentials securely
        creds = load_mongodb_credentials()
        
        # Create connection string
        connection_string = f"mongodb+srv://{creds['username']}:{creds['password']}@{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        
        # Create a new client and connect to the server
        client = MongoClient(connection_string)
        
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def clear_collection(db, collection_name):
    """Delete all documents in the collection"""
    try:
        result = db[collection_name].delete_many({})
        print(f"Deleted {result.deleted_count} documents from {collection_name}")
        return result.deleted_count
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return 0

def remove_duplicates(db, collection_name, field):
    """Remove duplicate documents based on a specific field"""
    try:
        collection = db[collection_name]
        
        # Find all unique values in the specified field
        pipeline = [
            {"$group": {
                "_id": f"${field}",
                "dups": {"$push": "$_id"},
                "count": {"$sum": 1}
            }},
            {"$match": {"count": {"$gt": 1}}
        }]
        
        duplicates = list(collection.aggregate(pipeline))
        
        if not duplicates:
            print(f"No duplicates found based on field: {field}")
            return 0
            
        print(f"Found {len(duplicates)} sets of duplicates based on field: {field}")
        
        # Keep the first occurrence and remove the rest
        removed_count = 0
        for dup in duplicates:
            # Keep the first document, delete the rest
            keep_id = dup["dups"][0]
            result = collection.delete_many({
                "_id": {"$in": dup["dups"][1:]}
            })
            removed_count += result.deleted_count
            
        print(f"Removed {removed_count} duplicate documents")
        return removed_count
        
    except Exception as e:
        print(f"Error removing duplicates: {e}")
        return 0

def insert_sample_data(db, collection_name, force_insert=False):
    try:
        collection = db[collection_name]
        
        # Check if collection has data
        if not force_insert and collection.count_documents({}) > 0:
            print("Collection already contains data. Use force_insert=True to insert anyway.")
            return []
            
        # Sample data to insert
        sample_data = [
            {"name": "John Doe", "age": 30, "email": "john@example.com"},
            {"name": "Jane Smith", "age": 25, "email": "jane@example.com"},
            {"name": "Bob Johnson", "age": 35, "email": "bob@example.com"}
        ]
        
        # Insert the data
        result = collection.insert_many(sample_data)
        print(f"Inserted {len(result.inserted_ids)} documents")
        return result.inserted_ids
    except Exception as e:
        print(f"Error inserting data: {e}")
        return None

def read_all_documents(db, collection_name):
    try:
        # Get the collection
        collection = db[collection_name]
        
        # Find all documents
        print("\nAll documents in the collection:")
        for doc in collection.find():
            pprint(doc)
    except Exception as e:
        print(f"Error reading data: {e}")

def main():
    # Connect to MongoDB
    client = connect_to_mongodb()
    
    if client:
        try:
            # Database and collection names
            db_name = "my_database"
            collection_name = "users"
            
            # Get the database (creates it if it doesn't exist)
            db = client[db_name]
            
            # First, remove duplicates based on name field
            print("Checking for and removing duplicate users...")
            removed = remove_duplicates(db, collection_name, "name")
            
            if removed > 0:
                print(f"Successfully removed {removed} duplicate documents")
            
            # Clear existing data (uncomment to clear the entire collection)
            # clear_collection(db, collection_name)
            
            # Insert sample data
            print("Inserting sample data...")
            insert_sample_data(db, collection_name, force_insert=False)
            
            # Read and display all documents
            read_all_documents(db, collection_name)
            
            # Example of querying a specific document
            print("\nQuerying for specific user:")
            user = db[collection_name].find_one({"name": "John Doe"})
            if user:
                print(f"Found user: {user['name']}, Age: {user['age']}, Email: {user['email']}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the connection when done
            client.close()
            print("\nConnection to MongoDB closed.")

if __name__ == "__main__":
    main()
