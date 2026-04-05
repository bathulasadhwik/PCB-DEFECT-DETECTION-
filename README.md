# PCB Detection System

A comprehensive PCB inspection system using AI and computer vision for defect detection and analysis.

## Features

- Real-time PCB inspection using YOLO models
- AI-powered defect analysis with Gemini integration
- Comprehensive reporting with PDF generation
- Web-based dashboard for analytics
- REST API for integration

## Tech Stack

- **Backend**: Python Flask, OpenCV, YOLO, Gemini AI
- **Frontend**: HTML/CSS/JavaScript
- **Database**: Local JSON logging
- **Deployment**: Railway (Backend), Firebase (Frontend)

## Local Development

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r backend/requirements.txt`
5. Run backend: `python backend/app.py`
6. Run frontend: `python -m http.server 5500` in frontend directory

## Deployment

### Frontend (Firebase Hosting)
```bash
firebase deploy --only hosting
```
URL: https://pcb-detection-820d6.web.app

### Backend (Railway)
1. Push code to GitHub
2. Connect repository to Railway
3. Set environment variables:
   - `PORT=8080`
   - `FLASK_ENV=production`
   - `GEMINI_API_KEY=your_key_here`

## API Endpoints

- `POST /api/inspect` - Perform PCB inspection
- `GET /api/analytics` - Get dashboard analytics
- `GET /reports/<filename>` - Download PDF reports

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key
- `YOLO_MODEL_KEY` - YOLO model selection
- `PORT` - Server port (default: 5000)