from pyspark.sql import Window,functions as F
from feature_engineering.analog_features import cosine
from feature_engineering.embedding_features import array_to_vector
from feature_engineering.observability import traced
@traced()
def build_top5_similar_products(q,h,cfg):
 a=q.select(F.col("product_id").alias("qid"),F.col("launch_date").alias("qd"),array_to_vector("product_embedding").alias("qv")); b=h.select(F.col("product_id").alias("aid"),F.col("product_name").alias("aname"),F.col("launch_date").alias("ad"),"m1_qty","m2_qty","m3_qty",array_to_vector("product_embedding").alias("av")); r=a.crossJoin(b).filter("ad<qd and m3_qty is not null").withColumn("similarity",cosine("qv","av")).withColumn("rank",F.row_number().over(Window.partitionBy("qid").orderBy(F.col("similarity").desc()))).filter(F.col("rank")<=cfg.top_k); return r.groupBy("qid").agg(F.sort_array(F.collect_list(F.struct("rank","aid","aname","similarity","ad","m1_qty","m2_qty","m3_qty"))).alias("top5_similar_products")).withColumnRenamed("qid","product_id")