from pyspark.sql import functions as F
from feature_engineering.common import accessory_expr
from feature_engineering.embedding_features import add_embeddings
from feature_engineering.observability import traced
@traced()
def build_product_history(fact,cfg):
 l=fact.groupBy("product_id").agg(F.min("order_date").alias("launch_date")); e=fact.join(l,"product_id").withColumn("m",F.floor(F.months_between("order_date","launch_date")).cast("int")+1)
 o=e.filter(F.col("m").between(1,3)).groupBy("product_id","launch_date").agg(*[F.sum(F.when(F.col("m")==i,F.col("quantity")).otherwise(0.0)).alias(f"m{i}_qty") for i in [1,2,3]],F.sum("quantity").alias("qty_90d"),F.countDistinct("customer_id").alias("customers_90d"),F.countDistinct("order_id").alias("orders_90d"),(F.sum("quantity")/F.greatest(F.countDistinct("order_id"),F.lit(1))).alias("avg_order_qty_90d"))
 p=fact.join(l,"product_id").groupBy("product_id","launch_date").agg(F.first("product_name",True).alias("product_name"),F.first("brand",True).alias("brand"),F.first("functional_category",True).alias("functional_category"),F.expr("percentile_approx(unit_price,.5)").alias("launch_price")).withColumn("refurbished_flag",F.when(F.col("product_name").rlike("refurbished|renewed"),1).otherwise(0)).withColumn("accessory_flag",accessory_expr(F.col("product_name"),F.col("functional_category")))
 return add_embeddings(p,cfg).join(o,["product_id","launch_date"],"left")