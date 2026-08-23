import copy, math, json, tempfile
from pathlib import Path
import numpy as np, torch, mlflow
from torch import nn
from torch.utils.data import Dataset,DataLoader
from training.neural_network import DemandNetwork
from feature_engineering.observability import traced

class ProductDataset(Dataset):
 def __init__(self,a): self.a=a
 def __len__(self): return len(self.a["target"])
 def __getitem__(self,i): return tuple(torch.tensor(self.a[k][i]) for k in ["embedding","numeric","brand","category","target"])

def criterion(name):
 return nn.MSELoss() if name=="mse" else nn.PoissonNLLLoss(log_input=True) if name=="poisson" else nn.HuberLoss(delta=1.0)

@traced()
def train_with_mlflow(train_arrays,val_arrays,preprocessor,cfg,train_cfg):
 mlflow.set_experiment(cfg.experiment_name)
 mlflow.set_registry_uri("databricks-uc")
 with mlflow.start_run(run_name="cold_start_nn") as run:
  mlflow.log_params({k:str(v) for k,v in train_cfg.__dict__.items()})
  mlflow.set_tags({"problem":"new_product_cold_start","targets":"m1,m2,m3","feature_table":f"{cfg.output_prefix}.product_features"})
  torch.manual_seed(train_cfg.seed); np.random.seed(train_cfg.seed)
  model=DemandNetwork(train_arrays["embedding"].shape[1],train_arrays["numeric"].shape[1],len(preprocessor["categorical_vocab"]["brand"]),len(preprocessor["categorical_vocab"]["functional_category"]),train_cfg)
  tr=DataLoader(ProductDataset(train_arrays),train_cfg.batch_size,shuffle=True); va=DataLoader(ProductDataset(val_arrays),train_cfg.batch_size)
  lossfn=criterion(train_cfg.loss); opt=torch.optim.AdamW(model.parameters(),lr=train_cfg.learning_rate,weight_decay=train_cfg.weight_decay)
  best,bstate,wait,hist=math.inf,None,0,[]
  for epoch in range(1,train_cfg.max_epochs+1):
   model.train(); tl=0.0
   for e,n,b,c,y in tr:
    opt.zero_grad(); loss=lossfn(model(e.float(),n.float(),b.long(),c.long()),y.float()); loss.backward(); opt.step(); tl+=loss.item()*len(y)
   model.eval(); vl=0.0
   with torch.no_grad():
    for e,n,b,c,y in va: vl+=lossfn(model(e.float(),n.float(),b.long(),c.long()),y.float()).item()*len(y)
   tl/=len(tr.dataset); vl/=len(va.dataset); hist.append({"epoch":epoch,"train_loss":tl,"validation_loss":vl})
   mlflow.log_metrics({"train_loss":tl,"validation_loss":vl},step=epoch); print(f"epoch={epoch:03d} train_loss={tl:.6f} validation_loss={vl:.6f}")
   if vl<best-1e-6: best,bstate,wait=vl,copy.deepcopy(model.state_dict()),0
   else:
    wait+=1
    if wait>=train_cfg.patience: break
  model.load_state_dict(bstate); mlflow.log_metric("best_validation_loss",best)
  with tempfile.TemporaryDirectory() as d:
   ck=Path(d)/"checkpoint.pt"; pp=Path(d)/"preprocessor.json"; hs=Path(d)/"loss_history.json"
   torch.save(bstate,ck); pp.write_text(json.dumps(preprocessor)); hs.write_text(json.dumps(hist)); mlflow.log_artifacts(d,"training_artifacts")
  return model,hist,run.info.run_id
