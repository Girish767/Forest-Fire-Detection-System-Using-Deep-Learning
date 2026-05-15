# Forest Fire Detection App

## Prerequisites
Ensure you have installed the required dependencies:
```bash
pip install -r requirements.txt
```
python version = 3.11.0

## Running the Application
To start the application, use the `streamlit run` command, not `python`:

```bash
python app.py
```

## Features
- **Image Analysis**: Upload satellite or aerial images to detect fire.
- **Video Analysis**: Upload videos to scan for fire in frames.
- **Settings**: Adjust confidence thresholds in the sidebar.

"Heavy Fire Detected in Video" - when it finds "Fire heavy" in any frame
"Moderate Fire Detected in Video" - when it finds "Fire moderate" or just "Fire"
"Smoke Detected in Video (Fire Risk)" - when it only finds smoke
"No Fire Detected" - when everything is normal