# Fraud Detection — CI/CD Project

**Author: Suresh D R | AI Product Developer & Technology Mentor**
*MLOps Syllabus — Deploy and Retrain ML Models on AWS*

## Quick Start

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cd src && python generate_data.py && python train.py && cd ..
pytest tests/test_model.py -v
streamlit run src/app.py
```

## CI/CD Pipeline

Every time you push code to GitHub — pipeline runs automatically:
- Installs Python
- Trains the model
- Runs all 8 tests
- Builds Docker image

## Project Structure

```
fraud-detection-cicd/
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml   ← CI/CD pipeline
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
├── tests/
│   └── test_model.py
├── Dockerfile
├── requirements.txt
└── README.md
```
