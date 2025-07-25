# IdentiFace - Face Recognition & Clustering Platform

A modern, dynamic face recognition and clustering system that processes group photos, extracts faces, generates embeddings, and groups similar faces together—similar to Google Photos' "People" feature.

## 🌟 Features

- **Face Detection & Extraction**: Automatically detect and extract faces from group photos
- **Face Clustering**: Group similar faces using advanced machine learning algorithms
- **Similarity Search**: Upload a photo to find similar faces in your collection
- **Dynamic UI**: Modern, responsive React interface with dark/light mode
- **Adjustable Similarity**: Real-time similarity threshold adjustment
- **People Gallery**: Browse all discovered people in an organized grid
- **Cluster View**: Detailed view of all photos for each person
- **Real-time Processing**: Fast face processing with progress indicators

## 🏗️ Architecture

### Frontend (React/Next.js)
- Modern, responsive UI with Tailwind CSS
- Dark/light mode support
- Framer Motion animations
- Real-time similarity filtering
- Drag & drop file uploads

### Backend (FastAPI)
- RESTful API with automatic documentation
- Face detection using MTCNN
- Face recognition using FaceNet (InceptionResnetV1)
- HDBSCAN clustering for grouping faces
- PostgreSQL database integration

### Database (PostgreSQL)
- Face metadata and embeddings storage
- Efficient similarity search with vector operations
- Upload logging and processing history

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- PostgreSQL 12+
- Git

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd face-recognition-copy/Identiface
   \`\`\`

2. **Set up the database**
   \`\`\`bash
   # Create PostgreSQL database
   createdb face_recognition_db
   
   # Run the SQL setup scripts
   psql -d face_recognition_db -f ../01_setup_extensions_and_tables.sql
   psql -d face_recognition_db -f ../02_create_indexes.sql
   psql -d face_recognition_db -f ../03_create_functions_and_triggers.sql
   \`\`\`

3. **Start the development environment**
   \`\`\`bash
   chmod +x start-dev.sh
   ./start-dev.sh
   \`\`\`

   This will:
   - Set up Python virtual environment
   - Install all dependencies
   - Start the FastAPI backend on port 8000
   - Start the Next.js frontend on port 3000

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📁 Project Structure

\`\`\`
Identiface/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main application page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── upload-zone.tsx    # File upload component
│   ├── people-gallery.tsx # People grid view
│   ├── search-results.tsx # Search results display
│   └── cluster-view.tsx   # Individual cluster view
├── lib/                   # Utilities and services
│   └── api-service.ts     # API client
├── backend/               # FastAPI backend
│   ├── main.py           # Main FastAPI application
│   ├── requirements.txt   # Python dependencies
│   └── start.sh          # Backend startup script
└── start-dev.sh          # Development environment starter
\`\`\`

## 🔧 Configuration

### Database Configuration
Update the database configuration in `backend/main.py`:

```python
DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': 5432
}
