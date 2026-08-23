# Databricks notebook source
# MAGIC %md
# MAGIC # New-product inference

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

from inference.feature_builder import create_inference_features_from_dict
from databricks.sdk import WorkspaceClient
import pandas as pd
from datetime import datetime
from pyspark.sql import Row

new_product={'product_id':'NEW_SKU_001','product_name':'Zamaha 7.2 Channel Wireless 4K AV Receiver Black','brand':'Zamaha','launch_price':599.0,'launch_date':'2026-09-01'}

# Build features
features=create_inference_features_from_dict(spark,new_product,CFG,as_of_date=new_product["launch_date"])
pdf=features.toPandas()

# Convert numpy arrays to lists for JSON serialization
if 'product_embedding' in pdf.columns:
    pdf['product_embedding']=pdf['product_embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)

# Prepare payload and predict
payload={"dataframe_split":{"columns":list(pdf.columns),"data":pdf.astype(object).where(pdf.notna(),None).values.tolist()}}
w=WorkspaceClient()
response=w.serving_endpoints.query(name=CFG.endpoint_name,dataframe_records=[payload])
predictions=response.predictions if hasattr(response,'predictions') else response

# Save predictions
rows=[]
for p in predictions:
    rows.append(Row(product_id=new_product["product_id"],predicted_at=datetime.utcnow(),endpoint=CFG.endpoint_name,prediction=p))
spark.createDataFrame(rows).write.mode("append").option("mergeSchema","true").format("delta").saveAsTable(CFG.prediction_table)

display(features)
print(predictions)