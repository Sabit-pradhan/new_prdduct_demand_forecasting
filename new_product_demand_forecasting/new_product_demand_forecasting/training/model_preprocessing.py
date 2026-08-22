import numpy as np
from pyspark.sql import functions as F
from config.config import FEATURE_COLUMNS
from feature_engineering.observability import traced
NUMERIC=[c for c in FEATURE_COLUMNS if c not in {"product_embedding","brand","functional_category"}]; TARGETS=["target_m1_qty","target_m2_qty","target_m3_qty"]
@traced()
def fit_preprocessor(d):
 r=d.agg(*[x for c in NUMERIC for x in [F.avg(c).alias(c+"_m"),F.stddev_pop(c).alias(c+"_s")]]).first().asDict(); v={};
 for c in ["brand","functional_category"]: v[c]={"__UNK__":0,**{x[c]:i+1 for i,x in enumerate(d.select(c).distinct().collect()) if x[c] is not None}}
 return {"numeric_columns":NUMERIC,"numeric_stats":{c:{"mean":float(r[c+"_m"] or 0),"std":max(float(r[c+"_s"] or 0),1e-6)} for c in NUMERIC},"categorical_vocab":v}
@traced()
def collect_numpy(d,a,target_transform="log1p"):
 p=d.select("product_id","product_embedding","brand","functional_category",*NUMERIC,*TARGETS).toPandas(); y=p[TARGETS].to_numpy(np.float32); y=np.log1p(np.maximum(y,0)) if target_transform=="log1p" else y; return {"product_id":p.product_id.to_numpy(),"embedding":np.stack(p.product_embedding.map(lambda x:np.asarray(x,np.float32))),"numeric":np.column_stack([(p[c].fillna(a["numeric_stats"][c]["mean"])-a["numeric_stats"][c]["mean"])/a["numeric_stats"][c]["std"] for c in NUMERIC]).astype(np.float32),"brand":p.brand.map(lambda x:a["categorical_vocab"]["brand"].get(x,0)).to_numpy(np.int64),"category":p.functional_category.map(lambda x:a["categorical_vocab"]["functional_category"].get(x,0)).to_numpy(np.int64),"target":y}