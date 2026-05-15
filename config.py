import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'fireguard-default-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    UPLOAD_FOLDER = 'static/uploads'
    
    # EmailJS Configuration (loaded from .env)
    EMAILJS_PUBLIC_KEY = os.getenv('EMAILJS_PUBLIC_KEY', '')
    EMAILJS_SERVICE_ID = os.getenv('EMAILJS_SERVICE_ID', '')
    EMAILJS_TEMPLATE_ID = os.getenv('EMAILJS_TEMPLATE_ID', '')
