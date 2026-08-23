from pyspark.sql import Window,functions as F
from feature_engineering.embedding_features import array_to_vector
from feature_engineering.observability import traced
@F.udf("double")
def cosine(a,b):
 if a is None or b is None:return None
 x,y=a.toArray(),b.toArray();d=float((x@x)**.5*(y@y)**.5);return float(x@y/d) if d else 0.0
@traced()
def build_analog_features(q,h,cfg):
 a=q.select(F.col("product_id").alias("qid"),F.col("launch_date").alias("qd"),F.col("brand").alias("qb"),F.col("functional_category").alias("qc"),array_to_vector("product_embedding").alias("qv")); b=h.select(F.col("launch_date").alias("hd"),F.col("brand").alias("hb"),F.col("functional_category").alias("hc"),"m1_qty","m2_qty","m3_qty","qty_90d","customers_90d","orders_90d","avg_order_qty_90d",array_to_vector("product_embedding").alias("hv"))
 r=a.crossJoin(b).filter("hd<qd and m3_qty is not null").withColumn("s",cosine("qv","hv")).withColumn("rn",F.row_number().over(Window.partitionBy("qid").orderBy(F.col("s").desc()))).filter(F.col("rn")<=cfg.top_k).withColumn("w",F.exp(F.lit(cfg.analog_temperature)*F.col("s")))
 z=r.groupBy("qid").agg(F.max("s").alias("top1_similarity"),F.avg("s").alias("top5_mean_similarity"),F.avg((F.col("qb")==F.col("hb")).cast("double")).alias("top5_same_brand_share"),F.avg((F.col("qc")==F.col("hc")).cast("double")).alias("top5_same_category_share"),F.sum("w").alias("ws"),*[F.sum(F.col("w")*F.col(c)).alias("x_"+c) for c in ["m1_qty","m2_qty","m3_qty","customers_90d","orders_90d","avg_order_qty_90d","qty_90d"]],F.sum(F.col("w")*F.pow("qty_90d",2)).alias("x2"))
 return z.select(F.col("qid").alias("product_id"),"top1_similarity","top5_mean_similarity","top5_same_brand_share","top5_same_category_share",(F.col("x_m1_qty")/F.col("ws")).alias("analog_weighted_m1_qty"),(F.col("x_m2_qty")/F.col("ws")).alias("analog_weighted_m2_qty"),(F.col("x_m3_qty")/F.col("ws")).alias("analog_weighted_m3_qty"),(F.col("x_customers_90d")/F.col("ws")).alias("analog_weighted_90d_customers"),(F.col("x_orders_90d")/F.col("ws")).alias("analog_weighted_90d_orders"),(F.col("x_avg_order_qty_90d")/F.col("ws")).alias("analog_weighted_avg_order_qty"),F.sqrt(F.greatest(F.col("x2")/F.col("ws")-F.pow(F.col("x_qty_90d")/F.col("ws"),2),F.lit(0.0))).alias("analog_demand_dispersion"))