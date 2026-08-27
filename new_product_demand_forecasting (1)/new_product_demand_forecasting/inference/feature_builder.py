from datetime import datetime
from pyspark.sql import functions as F
from feature_engineering.schemas import NEW_PRODUCT_SCHEMA
from feature_engineering.common import normalize_text,category_expr,accessory_expr
from feature_engineering.embedding_features import add_embeddings
from feature_engineering.analog_features import build_analog_features
from feature_engineering.point_in_time_features import build_point_in_time_features
from config.config import FEATURE_COLUMNS
def create_inference_features(spark,n,cfg,as_of_date=None):
 h=spark.table(f"{cfg.output_prefix}.product_history");f=spark.table(f"{cfg.output_prefix}.fact_order_line");q=n.withColumn("product_name",normalize_text("product_name")).withColumn("brand",normalize_text(F.coalesce("brand",F.element_at(F.split("product_name"," "),1)))).withColumn("functional_category",category_expr("product_name")).withColumn("refurbished_flag",F.when(F.col("product_name").rlike("refurbished|renewed"),1).otherwise(0)).withColumn("accessory_flag",accessory_expr(F.col("product_name"),F.col("functional_category")));q=add_embeddings(q,cfg);a=build_analog_features(q,h,cfg);p=build_point_in_time_features(q,h,f,cfg);x=q.join(a,"product_id","left").join(p,"product_id","left").withColumn("log_launch_price",F.log1p("launch_price")).withColumn("launch_month_sin",F.sin(F.lit(6.2831853)*F.month("launch_date")/12)).withColumn("launch_month_cos",F.cos(F.lit(6.2831853)*F.month("launch_date")/12));return x.select("product_id","product_name","launch_date","launch_price",*FEATURE_COLUMNS)
def create_inference_features_from_dict(spark,p,cfg,as_of_date=None):
 p=dict(p);p["launch_date"]=datetime.strptime(p["launch_date"],"%Y-%m-%d").date() if isinstance(p["launch_date"],str) else p["launch_date"];return create_inference_features(spark,spark.createDataFrame([p],NEW_PRODUCT_SCHEMA),cfg,as_of_date)