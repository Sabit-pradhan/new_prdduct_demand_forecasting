# Databricks notebook source
# MAGIC %md
# MAGIC # Final test evaluation

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

from training.model_loader import latest_run_id,load_training_run
run_id=latest_run_id(spark,CFG)
model,preprocessor=load_training_run(run_id,TRAINING_CFG)
test_df=spark.table(f'{CFG.output_prefix}.cold_start_test_features')
from evaluation.pipeline import run_evaluation
metrics,predictions,actual=run_evaluation(spark,model,preprocessor,run_id,CFG,TRAINING_CFG,test_df)
print(metrics)
