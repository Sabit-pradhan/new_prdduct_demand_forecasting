# Databricks notebook source
# MAGIC %md
# MAGIC # New-product inference

# COMMAND ----------

# MAGIC %pip install -U mlflow-skinny[databricks] sentence-transformers torch pandas numpy databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
PROJECT_ROOT='/Workspace/Repos/<user>/<repo>/cold_start_forecasting_final'
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from config.config import PIPELINE_CFG as CFG, TRAINING_CFG

# COMMAND ----------

from inference.predictor import predict_new_product
new_product={'product_id':'NEW_SKU_001','product_name':'Zamaha 7.2 Channel Wireless 4K AV Receiver Black','brand':'Zamaha','launch_price':599.0,'launch_date':'2026-09-01'}
features,predictions=predict_new_product(spark,new_product,CFG)
display(features)
print(predictions)