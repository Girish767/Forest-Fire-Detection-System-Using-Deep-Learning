from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
from db_models import db, User, DetectionLog
from utils import predict_image, process_video
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Make EmailJS config available in all templates
@app.context_processor
def inject_emailjs_config():
    return {
        'EMAILJS_PUBLIC_KEY': app.config.get('EMAILJS_PUBLIC_KEY', ''),
        'EMAILJS_SERVICE_ID': app.config.get('EMAILJS_SERVICE_ID', ''),
        'EMAILJS_TEMPLATE_ID': app.config.get('EMAILJS_TEMPLATE_ID', '')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('detect'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('register'))
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use a different email or login.')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_pw = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, email=email, password=hashed_pw, email_notifications=True)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if user wants email notification
        send_email = request.form.get('email_notification') == 'on'
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext in ['jpg', 'jpeg', 'png']:
                label, conf, scores = predict_image(filepath)
                
                # Log to DB
                new_log = DetectionLog(
                    detection_type=label,
                    confidence=conf,
                    media_type='image',
                    filename=filename
                )
                db.session.add(new_log)
                db.session.commit()
                
                return render_template('detect.html', result=label, confidence=conf, image=filename, scores=scores)
            
            elif file_ext in ['mp4', 'avi', 'mov']:
                detections = process_video(filepath, app.config['UPLOAD_FOLDER'])
                # Log significant detections (e.g., first fire frame)
                # Check for "Fire" or "Smoke" in the mapped label (case-insensitive)
                fire_detected = any("fire" in d['label'].lower() or "smoke" in d['label'].lower() for d in detections)
                
                if fire_detected:
                     # Find max confidence detection
                     max_conf_detection = max(detections, key=lambda x: x['confidence'])
                     
                     new_log = DetectionLog(
                        detection_type=max_conf_detection['label'],
                        confidence=max_conf_detection['confidence'],
                        media_type='video',
                        filename=filename
                    )
                     db.session.add(new_log)
                     db.session.commit()
                
                return render_template('detect.html', video_result=detections, video=filename)
                
    return render_template('detect.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('detect'))
    
    logs = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).all()
    return render_template('dashboard.html', logs=logs)

def create_admin():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            hashed_pw = generate_password_hash('admin123', method='scrypt')
            admin = User(username='admin', email='admin@fireguard.ai', password=hashed_pw, is_admin=True)
            db.session.add(admin)
            db.session.commit()
            print("Admin user created with email: admin@fireguard.ai")

if __name__ == '__main__':
    create_admin()
    app.run(debug=True)
