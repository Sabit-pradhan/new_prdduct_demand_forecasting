from datetime import datetime
import mlflow.deployments
from pyspark.sql import Row
from inference.feature_builder import create_inference_features_from_dict

def predict_new_product(spark,product,cfg):
 f=create_inference_features_from_dict(spark,product,cfg,as_of_date=product["launch_date"])
 pdf=f.toPandas(); payload={"dataframe_split":{"columns":list(pdf.columns),"data":pdf.astype(object).where(pdf.notna(),None).values.tolist()}}
 response=mlflow.deployments.get_deploy_client("databricks").predict(endpoint=cfg.endpoint_name,inputs=payload)
 predictions=response.get("predictions",response)
 rows=[]
 for p in predictions: rows.append(Row(product_id=product["product_id"],predicted_at=datetime.utcnow(),endpoint=cfg.endpoint_name,prediction=p))
 spark.createDataFrame(rows).write.mode("append").option("mergeSchema","true").format("delta").saveAsTable(cfg.prediction_table)
 return f,predictions
