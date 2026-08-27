# Module 6 — CI/CD Practical: Fraud Detection Project

**Author: Suresh D R | AI Product Developer & Technology Mentor**
*MLOps Syllabus — Deploy and Retrain ML Models on AWS*

---

## What You Will Do

```
Step 1  → Open project in VS Code
Step 2  → Create virtual environment
Step 3  → Install libraries
Step 4  → Generate data and train model
Step 5  → Run tests
Step 6  → Create GitHub repository
Step 7  → Create the CI/CD YAML file
Step 8  → Push everything to GitHub
Step 9  → Watch pipeline run automatically on GitHub
Step 10 → Make a code change on your laptop
Step 11 → Push the change to GitHub
Step 12 → Watch pipeline retrigger automatically
Step 13 → Compare old run vs new run on GitHub
```

---

## STEP 1 — Open Project in VS Code

1. Unzip `fraud-detection-github.zip` to Desktop
2. Open VS Code → **File** → **Open Folder**
3. Select `fraud-detection-github` → **Select Folder**

**Open Git Bash terminal:**
1. **Terminal** → **New Terminal**
2. Dropdown **(▾)** → **Select Default Profile** → **Git Bash**
3. **Terminal** → **New Terminal** again

**Confirm folder:**
```bash
pwd
```
```
/c/Users/Suresh D R/Desktop/fraud-detection-github
```

---

## STEP 2 — Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate
```

**You will see `(venv)` appear. ✅**

---

## STEP 3 — Install Libraries

```bash
pip install -r requirements.txt
```

Wait for all libraries to install. ✅

---

## STEP 4 — Generate Data and Train Model

```bash
cd src
python generate_data.py
python train.py
cd ..
```

**You will see:**
```
Created 1000 transactions
Fraud cases: 100 (10.0%) ✅

Accuracy  : 0.92
ROC-AUC   : 0.95
Model saved to: ../models/fraud_model.pkl ✅
```

---

## STEP 5 — Run Tests — All Must Pass

```bash
pytest tests/test_model.py -v
```

**You will see:**
```
tests/test_model.py::test_model_file_exists        PASSED
tests/test_model.py::test_model_loads              PASSED
tests/test_model.py::test_prediction_keys          PASSED
tests/test_model.py::test_prediction_is_binary     PASSED
tests/test_model.py::test_confidence_range         PASSED
tests/test_model.py::test_risk_level_valid         PASSED
tests/test_model.py::test_legitimate_prediction    PASSED
tests/test_model.py::test_fraud_prediction         PASSED

8 passed in 1.23s ✅
```

All 8 tests pass. ✅

---

## STEP 6 — Create GitHub Repository

1. Go to `https://github.com`
2. Click **+** → **New repository**
3. Name: `fraud-detection-cicd`
4. Set to **Public**
5. **Do NOT** tick README or .gitignore
6. Click **Create repository**
7. **Copy the URL:**
```
https://github.com/drsuresh8453/fraud-detection-cicd.git
```

---

## STEP 7 — Create the CI/CD YAML File

This is the most important step. This file tells GitHub Actions what to do automatically on every push.

### 7a — Create the folders

```bash
mkdir -p .github/workflows
```

**Verify:**
```bash
ls -la .github/
```
```
workflows/
```

### 7b — Create the YAML file

In VS Code left sidebar:
- Click on `.github` folder → click `workflows` folder
- Right click → **New File**
- Name it: `mlops_pipeline.yml`

**Paste this content into the file:**

```yaml
# .github/workflows/mlops_pipeline.yml
# This pipeline runs automatically every time you push code to main
# It installs Python, trains the model, runs all tests, builds Docker image

name: MLOps CI/CD Pipeline

on:
  push:
    branches: [main]       # triggers on every git push to main
  pull_request:
    branches: [main]       # triggers when PR is opened
  workflow_dispatch:        # allows manual trigger from GitHub UI

jobs:

  # ════════════════════════════════════════════
  # JOB 1 — TEST
  # Installs Python, trains model, runs all tests
  # If any test fails → pipeline stops here
  # Nothing gets built or deployed
  # ════════════════════════════════════════════
  test:
    name: Install, Train, Test
    runs-on: ubuntu-latest

    steps:
      - name: Step 1 — Checkout code from GitHub
        uses: actions/checkout@v3

      - name: Step 2 — Install Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Step 3 — Install all libraries
        run: pip install -r requirements.txt

      - name: Step 4 — Generate sample data
        run: |
          cd src
          python generate_data.py
          cd ..

      - name: Step 5 — Train the fraud detection model
        run: |
          cd src
          python train.py
          cd ..

      - name: Step 6 — Run all 8 pytest tests
        run: pytest tests/test_model.py -v

      - name: Step 7 — Show model file was created
        run: ls models/

  # ════════════════════════════════════════════
  # JOB 2 — BUILD DOCKER IMAGE
  # Only runs if Job 1 passes
  # Builds the complete Docker image
  # ════════════════════════════════════════════
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: test              # only runs if test job passes ✅

    steps:
      - name: Step 1 — Checkout code
        uses: actions/checkout@v3

      - name: Step 2 — Install Python and train model
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Step 3 — Install libraries and train
        run: |
          pip install -r requirements.txt
          cd src && python generate_data.py && python train.py && cd ..

      - name: Step 4 — Build Docker image
        run: |
          docker build -t fraud-detection:${{ github.sha }} .
          docker tag fraud-detection:${{ github.sha }} fraud-detection:latest

      - name: Step 5 — Verify Docker image was created
        run: docker images

      - name: Step 6 — Show success message
        run: |
          echo "========================================"
          echo "  Build Successful!"
          echo "  Image: fraud-detection:${{ github.sha }}"
          echo "  Commit by: ${{ github.actor }}"
          echo "  Branch: ${{ github.ref_name }}"
          echo "========================================"
```

**Save the file — `Ctrl+S`**

---

## STEP 8 — Push Everything to GitHub

### 8a — Initialise Git

```bash
git init
```

```
Initialized empty Git repository... ✅
```

### 8b — Configure Git (if not done before)

```bash
git config --global user.name "Suresh D R"
git config --global user.email "drsuresh8453@gmail.com"
```

### 8c — Check What Will Be Pushed

```bash
git status
```

**You will see all files listed — including `.github/workflows/mlops_pipeline.yml`** ✅

> ⚠️ data/ and models/ should NOT appear — .gitignore is protecting them.

### 8d — Stage All Files

```bash
git add .
git status
```

**All files in green — staged and ready.** ✅

### 8e — First Commit

```bash
git commit -m "Initial commit — fraud detection project with CI/CD pipeline"
```

**You will see:**
```
[master (root-commit) a1b2c3d] Initial commit — fraud detection project with CI/CD pipeline
 11 files changed, 350 insertions(+)
 create mode 100644 .github/workflows/mlops_pipeline.yml
 create mode 100644 Dockerfile
 create mode 100644 src/train.py
 ...
```

### 8f — Rename Branch and Connect to GitHub

```bash
git branch -M main
git remote add origin https://github.com/drsuresh8453/fraud-detection-cicd.git
git remote -v
```

```
origin  https://github.com/drsuresh8453/fraud-detection-cicd.git (fetch)
origin  https://github.com/drsuresh8453/fraud-detection-cicd.git (push)
```

### 8g — Push to GitHub

```bash
git push -u origin main
```

Enter your GitHub username and **Personal Access Token** as password.

**You will see:**
```
Enumerating objects: 13, done.
Writing objects: 100% (13/13), done.
To https://github.com/drsuresh8453/fraud-detection-cicd.git
 * [new branch]      main -> main ✅
```

---

## STEP 9 — Watch Pipeline Run Automatically on GitHub

The moment you pushed — **GitHub Actions triggered automatically.**

### 9a — Go to GitHub Actions

1. Open browser → go to:
```
https://github.com/drsuresh8453/fraud-detection-cicd
```
2. Click **Actions** tab (top menu)

**You will see your pipeline running:**
```
MLOps CI/CD Pipeline    ← spinning yellow circle = running
Initial commit — fraud detection project with CI/CD pipeline
```

### 9b — Click on the Running Pipeline

1. Click on the pipeline name
2. You will see the jobs:
```
Install, Train, Test    ← spinning = running
Build Docker Image      ← waiting (needs test to pass first)
```

### 9c — Click on the Test Job

You will see each step running one by one:
```
✅ Step 1 — Checkout code from GitHub
✅ Step 2 — Install Python 3.10
✅ Step 3 — Install all libraries
✅ Step 4 — Generate sample data
✅ Step 5 — Train the fraud detection model
✅ Step 6 — Run all 8 pytest tests
✅ Step 7 — Show model file was created
```

**Click on any step to see the exact terminal output inside.**

### 9d — Wait for Build Job

After test passes → Build job starts automatically:
```
✅ Step 1 — Checkout code
✅ Step 2 — Install Python
✅ Step 3 — Install libraries and train
✅ Step 4 — Build Docker image
✅ Step 5 — Verify Docker image
✅ Step 6 — Show success message
```

### 9e — Pipeline Complete

You will see:
```
✅ Install, Train, Test     2m 34s
✅ Build Docker Image       3m 12s

MLOps CI/CD Pipeline — SUCCESS ✅
```

**Your code is tested and Docker image is built — fully automatic!** 🎉

---

## STEP 10 — Make a Code Change on Your Laptop

> This shows the power of CI/CD. You make ONE change → push → pipeline retriggers automatically → new model tested and built.

**Back in VS Code — open `src/train.py`**

**Find this line:**
```python
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
```

**Change it to:**
```python
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
```

**Save — `Ctrl+S`**

**What changed:**
- `n_estimators` → 100 to 200 (more trees = more stable)
- `max_depth` → 5 to 10 (deeper trees = captures complex patterns)

---

## STEP 11 — Push the Change to GitHub

### 11a — See What Changed

```bash
git status
```

**You will see:**
```
On branch main
Changes not staged for commit:
        modified:   src/train.py
```

Only `train.py` is modified. ✅

```bash
git diff src/train.py
```

**You will see exactly what changed:**
```diff
- model = RandomForestClassifier(n_estimators=100, max_depth=5,
+ model = RandomForestClassifier(n_estimators=200, max_depth=10,
```

Green line = what you added.
Red line = what you removed.

### 11b — Stage, Commit, Push

```bash
git add src/train.py
git commit -m "Improve model — 200 trees, depth 10 for better accuracy"
git push
```

**You will see:**
```
Enumerating objects: 7, done.
Writing objects: 100% (4/4), done.
To https://github.com/drsuresh8453/fraud-detection-cicd.git
   a1b2c3d..b2c3d4e  main -> main ✅
```

---

## STEP 12 — Watch Pipeline Retrigger Automatically

**Go back to GitHub → Actions tab.**

**You will see a NEW pipeline run has started:**

```
MLOps CI/CD Pipeline    ← new run — spinning!
Improve model — 200 trees, depth 10 for better accuracy

MLOps CI/CD Pipeline    ← previous run (completed)
Initial commit — fraud detection project with CI/CD pipeline
```

**GitHub detected your new push and triggered the pipeline again — automatically.**
**You did not press anything.** 🎯

### What is Different This Time

```
FIRST RUN                    SECOND RUN
──────────────               ──────────────
n_estimators = 100           n_estimators = 200
max_depth = 5                max_depth = 10
commit: a1b2c3d              commit: b2c3d4e
image: fraud:a1b2c3d         image: fraud:b2c3d4e
```

GitHub Actions downloads YOUR NEW CODE (with 200 trees) — not the old code.
The model is trained with the new settings.
Tests run on the new model.
New Docker image built with the improved model inside.

---

## STEP 13 — Compare Old Run vs New Run on GitHub

### 13a — See Both Runs

On GitHub → **Actions** tab:

```
✅ Improve model — 200 trees, depth 10      ← run 2 (new)
✅ Initial commit — fraud detection project  ← run 1 (old)
```

### 13b — Click on Run 2 (New)

Click on **Step 5 — Train the fraud detection model**

**You will see in the logs:**
```
n_estimators=200, max_depth=10

  Accuracy  : 0.94    ← improved from 0.92!
  ROC-AUC   : 0.97    ← improved from 0.95!
```

### 13c — Click on Run 1 (Old) to Compare

Click on **Step 5 — Train the fraud detection model**

**You will see in the logs:**
```
n_estimators=100, max_depth=5

  Accuracy  : 0.92
  ROC-AUC   : 0.95
```

**You can clearly see:**
- Run 1 → old model → accuracy 92%
- Run 2 → new model → accuracy 94%
- Both are stored in GitHub Actions history forever

---

## What Just Happened — Summary

```
STEP 8  → You pushed code to GitHub
            Pipeline run #1 triggered automatically
            ✅ Tests passed
            ✅ Docker image built

STEP 11 → You changed ONE line in train.py
            You pushed to GitHub
            Pipeline run #2 triggered automatically
            ✅ New model trained with new settings
            ✅ Tests passed again
            ✅ New Docker image built with improved model
```

**You made a change → pushed → pipeline did everything else automatically.**

**This is CI/CD in action.** 🎯

---

## Key Things to Notice

| What | Why it matters |
|------|---------------|
| Pipeline triggered automatically | You did not press anything — just git push |
| Fresh machine every run | No leftover files from previous run |
| Model retrained on every push | Always latest code, latest model |
| Tests run on every push | Bad model never reaches production |
| Each run has unique SHA | You can always trace which code built which image |
| Full history on Actions tab | You can go back and see every run ever |

---

## Common Issues and Fixes

| Issue | Fix |
|-------|-----|
| Pipeline not showing on Actions tab | Wait 30 seconds and refresh |
| `No such file: mlops_pipeline.yml` | Check file is in `.github/workflows/` folder exactly |
| Tests fail on GitHub but pass locally | Check requirements.txt has all libraries |
| `git push` rejected | Run `git pull` first then push |
| YAML indentation error | Use 2 spaces everywhere — never Tab key |
| Docker build fails in pipeline | Check Dockerfile is in root folder |

---

## All Commands — Quick Reference

```bash
# Setup
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Run locally
cd src && python generate_data.py && python train.py && cd ..
pytest tests/test_model.py -v

# Create CI/CD pipeline file
mkdir -p .github/workflows
# create mlops_pipeline.yml inside .github/workflows/

# First push to GitHub
git init
git add .
git commit -m "Initial commit with CI/CD pipeline"
git branch -M main
git remote add origin https://github.com/username/repo.git
git push -u origin main

# Make a change and push again
# edit src/train.py
git add src/train.py
git commit -m "Improve model settings"
git push

# Watch on GitHub
# Go to → Actions tab → see pipeline running
```

---

*MLOps Syllabus — Deploy and Retrain ML Models on AWS*
*Author: Suresh D R | AI Product Developer & Technology Mentor*
