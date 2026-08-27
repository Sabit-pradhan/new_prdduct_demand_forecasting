import json,tempfile,os
from pathlib import Path
import mlflow,pandas as pd,torch
from mlflow.models import infer_signature
from model_wrapper import DemandPyFunc

def log_and_register(model,preprocessor,run_id,cfg,train_cfg,sample_pdf):
 mlflow.set_registry_uri("databricks-uc")
 # Get project root (this file is in register/deploy/)
 project_root = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
 code_files = [str(project_root/"register"/"deploy"/"model_wrapper.py"), str(project_root/"training")]
 with tempfile.TemporaryDirectory() as d, mlflow.start_run(run_id=run_id):
  d=Path(d); torch.save(model.state_dict(),d/"checkpoint.pt"); prep=dict(preprocessor); prep["embedding_example"]=sample_pdf.iloc[0].product_embedding; (d/"preprocessor.json").write_text(json.dumps(prep)); (d/"model_config.json").write_text(json.dumps(train_cfg.__dict__))
  example=sample_pdf.iloc[:1]; wrapper=DemandPyFunc();
  info=mlflow.pyfunc.log_model(name="model",python_model=wrapper,artifacts={"checkpoint":str(d/'checkpoint.pt'),"preprocessor":str(d/'preprocessor.json'),"model_config":str(d/'model_config.json')},input_example=example,pip_requirements=["mlflow","torch","pandas","numpy"],code_paths=code_files,registered_model_name=cfg.model_fqn)
  return info.registered_model_version
