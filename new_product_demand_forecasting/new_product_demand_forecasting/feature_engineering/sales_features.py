from pyspark.sql import functions as F
from feature_engineering.schemas import PRODUCT_STRUCT
from feature_engineering.common import normalize_text,category_expr
from feature_engineering.observability import traced
@traced()
def order_lines_from_sales(spark,cfg):
 r=spark.table(cfg.sales_fqn).withColumn("p",F.from_json(F.col("product").cast("string"),PRODUCT_STRUCT))
 return r.select(F.sha2(F.concat_ws("||","customer_id","order_date",F.col("p.id")),256).alias("order_id"),F.col("customer_id").cast("string").alias("customer_id"),F.to_timestamp("order_date").alias("order_ts"),F.to_date("order_date").alias("order_date"),F.col("p.id").alias("product_id"),normalize_text("product_name").alias("product_name"),normalize_text("product_category").alias("brand"),category_expr("product_name").alias("functional_category"),F.col("p.price").cast("double").alias("unit_price"),F.col("p.qty").cast("double").alias("quantity"),F.col("p.curr").alias("currency"),F.col("p.unit").alias("unit"),F.lit(0).alias("promotion_flag"),F.lit(0.0).alias("promotion_discount"),F.lit("sales").alias("source")).filter("product_id is not null and quantity>0").dropDuplicates(["order_id","product_id"])