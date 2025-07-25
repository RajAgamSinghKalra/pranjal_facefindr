# IdentiFace Backend v2.0.0

A modern, lightweight FastAPI backend for the IdentiFace face recognition platform.

## Features

- ✅ **Zero Dependencies Conflicts** - Minimal, clean dependencies
- ✅ **Mock Data Generation** - Realistic demo data for development
- ✅ **Full API Coverage** - All endpoints needed by the frontend
- ✅ **Error Handling** - Comprehensive error handling and logging
- ✅ **Auto Documentation** - Interactive API docs at `/docs`
- ✅ **Health Monitoring** - Health check endpoints
- ✅ **CORS Support** - Configured for frontend integration

## Quick Start

### 1. Setup
\`\`\`bash
cd backend
chmod +x start.sh
./start.sh
\`\`\`

### 2. Manual Setup
\`\`\`bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

## API Endpoints

### Core Endpoints
- `GET /` - API health check
- `GET /health` - Detailed health information
- `GET /docs` - Interactive API documentation

### Face Recognition API
- `GET /api/stats` - Get application statistics
- `GET /api/clusters` - Get all face clusters
- `GET /api/clusters/{id}` - Get specific cluster
- `POST /api/search` - Search for similar faces
- `POST /api/upload` - Upload and process photos
- `POST /api/process` - Batch process photos
- `PUT /api/similarity-threshold` - Update similarity threshold

### System API
- `GET /api/system/info` - Get system information

## Configuration

The backend uses minimal configuration stored in `config.json`:

\`\`\`json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "mock_data": {
    "clusters": 15,
    "min_faces_per_cluster": 3,
    "max_faces_per_cluster": 12
  }
}
\`\`\`

## Development

### Project Structure
\`\`\`
backend/
├── main.py           # Main FastAPI application
├── models.py         # Pydantic models
├── utils.py          # Utility functions
├── requirements.txt  # Dependencies
├── config.json       # Configuration
├── start.sh         # Startup script
└── README.md        # This file
\`\`\`

### Adding Features

1. **New Endpoints**: Add to `main.py`
2. **Data Models**: Define in `models.py`
3. **Utilities**: Add to `utils.py`
4. **Configuration**: Update `config.json`

## Testing

\`\`\`bash
# Test basic functionality
curl http://localhost:8000/
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/clusters

# View interactive docs
open http://localhost:8000/docs
\`\`\`

## Troubleshooting

### Port Already in Use
\`\`\`bash
lsof -ti:8000 | xargs kill -9
\`\`\`

### Dependencies Issues
\`\`\`bash
# Clean install
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

### Check Logs
The application logs all activities. Check console output for detailed information.

## Production Deployment

For production deployment, consider:

1. **Environment Variables** for configuration
2. **Database Integration** (PostgreSQL, MongoDB)
3. **Authentication & Authorization**
4. **Rate Limiting**
5. **Monitoring & Logging**
6. **Docker Containerization**

## License

MIT License - See LICENSE file for details.
\`\`\`
