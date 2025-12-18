import pymongo

def check_databases():
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    
    try:
        # List all databases
        print("\n=== Available Databases ===")
        for db_name in client.list_database_names():
            print(f"\nDatabase: {db_name}")
            db = client[db_name]
            
            # List collections in the database
            collections = db.list_collection_names()
            if collections:
                print("  Collections:")
                for collection in collections:
                    count = db[collection].count_documents({})
                    print(f"    - {collection} (Documents: {count})")
            else:
                print("  No collections")
                
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_databases()
