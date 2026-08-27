import math
from pyspark.sql import functions as F
from feature_engineering.sales_features import order_lines_from_sales
from feature_engineering.sales_order_features import order_lines_from_sales_order
from feature_engineering.product_history import build_product_history
from feature_engineering.analog_features import build_analog_features
from feature_engineering.point_in_time_features import build_point_in_time_features
from feature_engineering.similar_product_features import build_top5_similar_products
from config.config import FEATURE_COLUMNS
from feature_engineering.observability import traced

@traced(*FEATURE_COLUMNS)
def create_and_save_features(spark,cfg):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.tgt_catalog}.{cfg.feature_schema}")
    if cfg.source_mode=="sales_order": fact=order_lines_from_sales_order(spark,cfg)
    elif cfg.source_mode=="sales": fact=order_lines_from_sales(spark,cfg)
    elif cfg.source_mode=="union": fact=order_lines_from_sales_order(spark,cfg).unionByName(order_lines_from_sales(spark,cfg),allowMissingColumns=True).dropDuplicates(["order_id","product_id","unit_price","quantity"])
    else: raise ValueError("source_mode must be sales_order, sales, or union")
    fact.write.mode("overwrite").format("delta").saveAsTable(f"{cfg.output_prefix}.fact_order_line")
    history=build_product_history(fact,cfg)
    history.write.mode("overwrite").format("delta").saveAsTable(f"{cfg.output_prefix}.product_history")
    analog=build_analog_features(history,history,cfg)
    pit=build_point_in_time_features(history,history,fact,cfg)
    top5=build_top5_similar_products(history,history,cfg)
    top5.write.mode("overwrite").format("delta").saveAsTable(f"{cfg.output_prefix}.top5_similar_products")
    x=(history.join(analog,"product_id","left").join(pit,"product_id","left")
       .withColumn("log_launch_price",F.log1p(F.greatest("launch_price",F.lit(0.0))))
       .withColumn("launch_month_sin",F.sin(F.lit(2*math.pi)*F.month("launch_date")/12))
       .withColumn("launch_month_cos",F.cos(F.lit(2*math.pi)*F.month("launch_date")/12))
       .withColumnRenamed("m1_qty","target_m1_qty").withColumnRenamed("m2_qty","target_m2_qty").withColumnRenamed("m3_qty","target_m3_qty"))
    for c in [z for z in FEATURE_COLUMNS if z not in {"product_embedding","brand","functional_category"}]: x=x.withColumn(c,F.coalesce(F.col(c).cast("double"),F.lit(0.0)))
    x=x.select("product_id","product_name","launch_date","launch_price",*FEATURE_COLUMNS,"target_m1_qty","target_m2_qty","target_m3_qty").filter(F.col("target_m3_qty").isNotNull())
    x.write.mode("overwrite").format("delta").saveAsTable(f"{cfg.output_prefix}.product_features")
    return x,top5
