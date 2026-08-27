# Databricks notebook source
# MAGIC %md
# MAGIC # Conditional registration and deployment

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
import importlib.util,sys,importlib
sys.path.insert(0,PROJECT_ROOT+'/register/deploy')
import deployer
importlib.reload(deployer)
from deployer import should_deploy,deploy_version
from registry import log_and_register
from training.model_preprocessing import NUMERIC
sample=spark.table(f'{CFG.output_prefix}.cold_start_train_features').select('product_embedding','brand','functional_category',*NUMERIC).limit(5).toPandas()
sample['product_embedding']=sample['product_embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
latest=spark.table(CFG.evaluation_table).orderBy('evaluated_at',ascending=False).first()

# Get deployment decision with detailed comparison
should_deploy_flag,comparison_info=should_deploy(CFG,latest.run_id,latest.selection_metric_value)

# Display comparison details
print("="*80)
print("MODEL DEPLOYMENT EVALUATION")
print("="*80)
print(f"\nModel Name: {comparison_info['model_name']}")
print(f"Metric: {comparison_info['metric_name']}")
print(f"\n--- CANDIDATE MODEL ---")
print(f"Run ID: {comparison_info['candidate_run_id']}")
print(f"Metric Value: {comparison_info['candidate_metric']:.4f}")

if comparison_info['champion_exists']:
    print(f"\n--- CHAMPION MODEL ---")
    print(f"Version: {comparison_info['champion_version']}")
    print(f"Run ID: {comparison_info['champion_run_id']}")
    champion_metric_str = f"{comparison_info['champion_metric']:.4f}" if comparison_info['champion_metric'] is not None else 'N/A'
    print(f"Metric Value: {champion_metric_str}")
    print(f"\n--- COMPARISON ---")
    print(comparison_info['comparison'])
    if comparison_info.get('improvement_pct') is not None:
        print(f"Improvement: {comparison_info['improvement_pct']:+.2f}%")
else:
    print(f"\n--- CHAMPION MODEL ---")
    print("No existing champion model found")

print(f"\n--- DECISION ---")
print(f"Deploy: {comparison_info['decision']}")
print(f"Reason: {comparison_info['reason']}")
print("="*80)

# Deploy if decision is True
if should_deploy_flag:
    print("\n✓ Proceeding with deployment...")
    version=log_and_register(model,preprocessor,latest.run_id,CFG,TRAINING_CFG,sample)
    endpoint=deploy_version(CFG,version)
    print(f"\n✓ Successfully deployed version {version} to endpoint: {endpoint}")
    print(f"✓ Model registered as 'champion' alias")
else:
    print(f"\n✗ Deployment skipped - candidate did not beat champion")
    print(f"✗ Current champion version {comparison_info['champion_version']} remains active")