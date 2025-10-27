## Fake News Detection Service

A service for detecting fake news using text classification with FastAPI.


### Dataset:
You can download the fake news + real news dataset from the provided link or directly from the repository files

Fake News Detection - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv


### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Setup Instructions

You can set up your Python environment using either uv (Recommended) or standard venv with pip.


### Step 1. Create and Activate Virtual Environment

#### Option 1. Using uv (Recommended) - https://docs.astral.sh/uv/getting-started/installation/
I recommend uv for faster dependency resolution and built-in virtual environment management
```bash
# Install uv
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

#### Option 2. Using venv + pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 2. Install dependencies

If you are using uv:

```bash
uv pip install -r requirements.txt
```

If you are using pip:

```bash
pip install -r requirements.txt
```

### Step 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Step 4. Prepare Data

Ensure `fake.csv` and `true.csv` are in the root directory (same folder as `requirements.txt`)

## Training the Model

Run the training script to train and save the model:

```bash
python scripts/train.py
```

This will load and preprocess the data from fake.csv and true.csv, trains a Logistic Regression model using TF-IDF vectorization, saves the model and vectorizer to the models/ directory, and outputs accuracy metrics along with a classification report.

### Step 5. Running the API Server

### Development Mode

```bash
python -m fake_news_service.api
```

### Production Mode

```bash
python -m uvicorn fake_news_service.api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Usage

### Interactive Documentation

- **Swagger Docs UI**: `http://localhost:8000/docs`

### Predict Endpoint

**POST** `/predict`

**Request Body:**
```json
{
    "text": "news article."
}
```

**Response:**
```json
{
    "prediction": "FAKE"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Any sample news article to classify."}'
```

### Testing

Run the tests with verbose output:

```bash
python -m pytest tests/ -v
```

Run specific test:
```bash
python -m pytest tests/test_utils.py::TestUtils::test_clean_text_basic -v
```


### Development

#### Code Organization

- `data_loader.py`: Handles CSV loading and text preprocessing
- `model.py`: Contains model training and evaluation logic
- `api.py`: FastAPI application with prediction endpoints
- `utils.py`: Contains functions for text cleaning and model operations
- `train.py`: Training script


#### Common Issues

1. **Model not found error**: Run `python scripts/train.py` first
2. **NLTK data missing**: Run the NLTK download command (revisit Step 3 in Setup Instructions)
3. **Port already in use**: Change port in `api.py` or kill existing process

### v1.0.0
Initial release of a FastAPI-based fake news detection service using Logistic Regression with TF-IDF vectorization