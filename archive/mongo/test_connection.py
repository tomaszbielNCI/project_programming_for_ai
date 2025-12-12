import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure


def print_safe(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


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
        print_safe(f"Error loading MongoDB credentials: {e}")
        raise


def test_connection():
    """Test MongoDB connection and return client if successful"""
    try:
        # Load credentials
        creds = load_mongodb_credentials()
        
        # Create connection string
        connection_string = f"mongodb+srv://{creds['username']}:{creds['password']}@{creds['cluster']}/?retryWrites=true&w=majority&appName=Cluster0"
        
        # Create a new client and connect to the server
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print_safe("✅ Successfully connected to MongoDB!")
        
        # Get server info
        server_info = client.server_info()
        print_safe("\n=== MongoDB Server Information ===")
        print_safe(f"Version: {server_info.get('version')}")
        print_safe(f"Host: {client.HOST}")
        print_safe(f"Port: {client.PORT}")
        
        # List databases
        print_safe("\n=== Available Databases ===")
        for db_name in client.list_database_names():
            print_safe(f"- {db_name}")
        
        return client
        
    except OperationFailure as e:
        print_safe("\n❌ Authentication failed:")
        print_safe(f"{e}")
        print_safe("\nPlease check your username and password in mongo_password.txt")
    except ServerSelectionTimeoutError as e:
        print_safe("\n❌ Connection timed out:")
        print_safe(f"{e}")
        print_safe("\nPlease check your internet connection and IP whitelist in MongoDB Atlas")
    except Exception as e:
        print_safe(f"\n❌ Error connecting to MongoDB: {e}")
    
    return None


class TestMongoDBConnection(unittest.TestCase):
    @patch('test_connection.print_safe')
    @patch('test_connection.load_mongodb_credentials')
    @patch('pymongo.MongoClient')
    def test_connection_success(self, mock_client, mock_creds, mock_print):
        """Test successful MongoDB connection"""
        # Setup mock credentials
        test_creds = {
            'username': 'test_user',
            'password': 'test_pass',
            'cluster': 'test-cluster',
            'database': 'test_db'
        }
        mock_creds.return_value = test_creds
        
        # Create a mock client instance
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Setup mock admin and its methods
        mock_admin = MagicMock()
        mock_instance.admin = mock_admin
        mock_admin.command.return_value = {'ok': 1}
        
        # Setup mock server info
        mock_instance.server_info.return_value = {'version': '5.0.0'}
        mock_instance.HOST = 'test-host'
        mock_instance.PORT = 27017
        mock_instance.list_database_names.return_value = ['test_db']
        
        # Call the function
        with patch('test_connection.MongoClient', mock_client):
            client = test_connection()
        
        # Verify the connection was attempted with correct parameters
        mock_client.assert_called_once()
        call_args, call_kwargs = mock_client.call_args
        self.assertIn('serverSelectionTimeoutMS', call_kwargs)
        self.assertEqual(call_kwargs['serverSelectionTimeoutMS'], 10000)
        
        # Verify the client was used correctly
        mock_admin.command.assert_called_once_with('ping')
        
        # Verify success message was printed
        mock_print.assert_any_call("✅ Successfully connected to MongoDB!")
    
    @patch('test_connection.print_safe')
    @patch('test_connection.load_mongodb_credentials')
    @patch('pymongo.MongoClient')
    def test_connection_failure(self, mock_client, mock_creds, mock_print):
        """Test MongoDB connection failure"""
        # Setup mock credentials
        test_creds = {
            'username': 'test_user',
            'password': 'wrong_pass',
            'cluster': 'test-cluster',
            'database': 'test_db'
        }
        mock_creds.return_value = test_creds
        
        # Setup mock to raise connection error
        error_msg = "Connection failed"
        mock_client.return_value.admin.command.side_effect = ServerSelectionTimeoutError(error_msg)
        
        # Call the function
        with patch('test_connection.MongoClient', mock_client):
            client = test_connection()
        
        # Assertions
        self.assertIsNone(client)
        
        # Verify error handling
        mock_print.assert_any_call("\n❌ Connection timed out:")
        mock_print.assert_any_call(error_msg)
        mock_print.assert_any_call("\nPlease check your internet connection and IP whitelist in MongoDB Atlas")


if __name__ == "__main__":
    # Run the actual connection test
    print_safe("=== Testing MongoDB Connection ===")
    client = test_connection()
    if client:
        client.close()
    
    # Run unit tests if explicitly requested
    if '--test' in sys.argv:
        print_safe("\n=== Running Unit Tests ===")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Keep window open on Windows if not running tests
    elif sys.platform == "win32":
        input("\nPress Enter to exit...")
