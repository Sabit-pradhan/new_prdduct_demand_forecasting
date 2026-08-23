import json, numpy as np, pandas as pd, torch, mlflow
from training.neural_network import DemandNetwork
class DemandPyFunc(mlflow.pyfunc.PythonModel):
 def load_context(self,context):
  self.prep=json.load(open(context.artifacts["preprocessor"])); self.cfg=json.load(open(context.artifacts["model_config"]))
  class C: pass
  c=C()
  for k,v in self.cfg.items(): setattr(c,k,tuple(v) if k=="hidden_dims" else v)
  self.model=DemandNetwork(len(self.prep["embedding_example"]),len(self.prep["numeric_columns"]),len(self.prep["categorical_vocab"]["brand"]),len(self.prep["categorical_vocab"]["functional_category"]),c)
  self.model.load_state_dict(torch.load(context.artifacts["checkpoint"],map_location="cpu")); self.model.eval()
 def predict(self,context,model_input,params=None):
  emb=np.stack(model_input.product_embedding.map(lambda x: np.asarray(x,dtype=np.float32)))
  num=np.column_stack([(model_input[c].fillna(self.prep["numeric_stats"][c]["mean"]).to_numpy(np.float32)-self.prep["numeric_stats"][c]["mean"])/self.prep["numeric_stats"][c]["std"] for c in self.prep["numeric_columns"]]).astype(np.float32)
  brand=model_input.brand.map(lambda x:self.prep["categorical_vocab"]["brand"].get(x,0)).to_numpy(np.int64); cat=model_input.functional_category.map(lambda x:self.prep["categorical_vocab"]["functional_category"].get(x,0)).to_numpy(np.int64)
  with torch.no_grad(): p=self.model(torch.tensor(emb),torch.tensor(num),torch.tensor(brand),torch.tensor(cat)).numpy()
  if self.cfg.get("target_transform")=="log1p": p=np.maximum(np.expm1(p),0)
  return pd.DataFrame({"prediction_m1":p[:,0],"prediction_m2":p[:,1],"prediction_m3":p[:,2],"prediction_3m_total":p.sum(1)})
