from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    email_notifications = db.Column(db.Boolean, default=True)

class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    detection_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    media_type = db.Column(db.String(20), nullable=False) # 'image' or 'video'
    filename = db.Column(db.String(100), nullable=True)
