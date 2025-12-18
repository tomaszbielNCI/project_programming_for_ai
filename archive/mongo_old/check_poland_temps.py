import pymongo
import pandas as pd
from pprint import pprint

def check_temperature_data():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        
        # List all databases to check which one has our data
        print("\nAvailable databases:")
        for db_name in client.list_database_names():
            if db_name not in ['admin', 'local', 'config']:  # Skip system databases
                print(f"\nDatabase: {db_name}")
                db = client[db_name]
                collections = db.list_collection_names()
                for collection_name in collections:
                    print(f"  Collection: {collection_name}")
                    
                    # Check if collection has temperature data
                    sample = db[collection_name].find_one({"country": "Poland"})
                    if sample:
                        print("\nSample data for Poland:")
                        pprint(sample)
                        
                        # Get all temperature data for Poland
                        poland_data = list(db[collection_name].find({"country": "Poland"}).sort("year", 1))
                        
                        if poland_data:
                            print("\nAll temperature data for Poland:")
                            for doc in poland_data:
                                print(f"Year: {doc.get('year')}, Temp: {doc.get('avg_temp')}°C")
                            
                            # Calculate statistics
                            temps = [d.get('avg_temp', 0) for d in poland_data if 'avg_temp' in d]
                            if temps:
                                print(f"\nStatistics for Poland:")
                                print(f"Min temp: {min(temps):.2f}°C")
                                print(f"Max temp: {max(temps):.2f}°C")
                                print(f"Avg temp: {sum(temps)/len(temps):.2f}°C")
                        
                        return  # Found the data, no need to check other collections
        
        print("\nNo temperature data found for Poland in any collection.")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    print("Checking temperature data for Poland...")
    check_temperature_data()
