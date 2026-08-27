from pyspark.sql import functions as F,types as T
from pyspark.ml.linalg import Vectors,VectorUDT
from feature_engineering.observability import traced
@traced("product_embedding")
def add_embeddings(df,cfg,text_col="product_name"):
 @F.pandas_udf(T.ArrayType(T.FloatType()))
 def enc(s):
  import os
  import tempfile
  cache_dir = tempfile.mkdtemp()
  os.environ['TRANSFORMERS_CACHE']=cache_dir
  os.environ['HF_HOME']=cache_dir
  os.environ['SENTENCE_TRANSFORMERS_HOME']=cache_dir
  from sentence_transformers import SentenceTransformer
  import pandas as pd
  m=SentenceTransformer(cfg.embedding_model)
  return pd.Series([x.astype("float32").tolist() for x in m.encode(s.fillna("").tolist(),normalize_embeddings=True)])
 return df.withColumn("product_embedding",enc(text_col))
@F.udf(VectorUDT())
def array_to_vector(a): return Vectors.dense(a) if a is not None else None