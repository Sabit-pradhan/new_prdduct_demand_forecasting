from pyspark.sql import Window,functions as F
from feature_engineering.observability import traced
@traced()
def add_chronological_split(df,c):
 d=df.select("launch_date").distinct().withColumn("r",F.percent_rank().over(Window.orderBy("launch_date"))).withColumn("dataset_split",F.when(F.col("r")<c.train_fraction,"train").when(F.col("r")<c.train_fraction+c.validation_fraction,"validation").otherwise("test")); return df.join(d.select("launch_date","dataset_split"),"launch_date")
@traced()
def persist_splits(df,cfg):
 for s in ["train","validation","test"]: df.filter(F.col("dataset_split")==s).write.mode("overwrite").format("delta").saveAsTable(f"{cfg.output_prefix}.cold_start_{s}_features")
 return df