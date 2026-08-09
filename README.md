# AI Fake News Detector

A comprehensive AI-powered application designed to detect and analyze fake news using machine learning and natural language processing techniques.

<img width="1536" height="2048" alt="fakenewsdetectorbyabdullah929 netlify app_(iPad Mini)" src="https://github.com/user-attachments/assets/432d9d78-3944-4fd5-89a6-2ff6a5792fc2" />


## 🚀 Features

- **Real-time News Analysis**: Analyze news articles and content for authenticity
- **Machine Learning Models**: State-of-the-art AI models for fake news detection
- **Interactive Web Interface**: User-friendly frontend for easy interaction
- **RESTful API Backend**: Robust backend API for integration with external systems
- **High Accuracy**: Advanced NLP techniques for improved detection accuracy

## 📋 Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Tech Stack

- **Frontend**: JavaScript, HTML, CSS
  - Interactive user interface for news analysis
  - Real-time results display
  
- **Backend**: Python
  - Flask/FastAPI for REST API
  - Machine learning model deployment
  - Natural language processing with NLP libraries

- **Language Composition**:
  - JavaScript: 44%
  - Python: 38.9%
  - HTML: 11.1%
  - CSS: 6%

## 📁 Project Structure

```
AI-FAKE-NEWS-DETECTOR/
├── backend/              # Python backend server
│   ├── models/          # ML models for fake news detection
│   ├── api/             # API endpoints
│   └── utils/           # Utility functions
├── frontend/            # JavaScript/HTML/CSS frontend
│   ├── index.html       # Main page
│   ├── styles/          # CSS stylesheets
│   └── scripts/         # JavaScript files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- Node.js (optional, for frontend development)
- pip (Python package manager)

### Setup Backend

1. Clone the repository:
```bash
git clone https://github.com/Abdullah929-design/AI-FAKE-NEWS-DETECTOR.git
cd AI-FAKE-NEWS-DETECTOR
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to backend directory:
```bash
cd backend
```

4. Run the backend server:
```bash
python app.py
```

The backend will be available at `http://localhost:5000` (or your configured port)

### Setup Frontend

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Open `index.html` in your web browser or serve with a local server:
```bash
python -m http.server 8000
```

3. Access the application at `http://localhost:8000`

## 🎯 Usage

1. **Submit News Content**: Enter a news article or text in the input field
2. **Analyze**: Click the analyze button to process the content
3. **View Results**: Get detailed authenticity scores and analysis
4. **Review Metrics**: Check confidence scores and supporting details

## 🔌 API Documentation

### Analyze News Article

**Endpoint**: `POST /api/analyze`

**Request Body**:
```json
{
  "text": "News article content here...",
  "url": "https://example.com/article" (optional)
}
```

**Response**:
```json
{
  "is_fake": boolean,
  "confidence": float (0-1),
  "analysis": {
    "credibility_score": float,
    "key_findings": [],
    "sources": []
  }
}
```

### Get Supported Languages

**Endpoint**: `GET /api/languages`

**Response**:
```json
{
  "languages": ["en", "es", "fr", "de", "ar"]
}
```

## 📚 Dependencies

Core dependencies (see `requirements.txt`):
- Machine Learning frameworks (TensorFlow, PyTorch, scikit-learn)
- NLP libraries (NLTK, spaCy)
- Web framework (Flask/FastAPI)
- Data processing (pandas, numpy)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Support

For questions or issues, please:
- Open an issue on GitHub
- Check existing documentation
- Review the API documentation above

---

**Created by**: Abdullah929-design  
**Last Updated**: November 2025
