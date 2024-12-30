# config.py
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database configuration
DB_URI = os.getenv('DB_URI')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Default graph parameters
DEFAULT_K = 30
DEFAULT_DEPTH = 2
DIMENSION = 1536  # Dimension of the embeddings

BATCH_SIZE = 1000

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')