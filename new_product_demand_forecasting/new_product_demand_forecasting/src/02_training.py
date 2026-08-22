# Databricks notebook source
# MAGIC %md
# MAGIC # Training and MLflow tracking

# COMMAND ----------

# MAGIC %pip install -U mlflow-skinny[databricks] sentence-transformers torch pandas numpy databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

# 1. Get the current absolute notebook path
raw_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# 2. Split the path into parts
parts = raw_path.split('/')

# 3. Reconstruct up to the repo level (Parts: 0='', 1='Repos', 2='<user>', 3='<repo>')
repo_root = "/".join(parts[:4])

# 4. Prepend '/Workspace'
workspace_root_path = f"/Workspace{repo_root}"

print(workspace_root_path)

# COMMAND ----------

import sys
PROJECT_ROOT=workspace_root_path
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from config.config import PIPELINE_CFG as CFG, TRAINING_CFG

# COMMAND ----------

from training.pipeline import run_training
model,preprocessor,loss_history,run_id,test_df=run_training(spark,CFG,TRAINING_CFG)
print('MLflow run:',run_id)
display(spark.createDataFrame(loss_history))