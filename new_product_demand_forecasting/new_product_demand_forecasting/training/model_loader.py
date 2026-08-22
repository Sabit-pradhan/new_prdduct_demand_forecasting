import json, mlflow, torch
from pathlib import Path
from training.neural_network import DemandNetwork

def load_training_run(run_id, train_cfg):
    local=Path(mlflow.artifacts.download_artifacts(run_id=run_id,artifact_path="training_artifacts"))
    prep=json.loads((local/"preprocessor.json").read_text())
    model=DemandNetwork(prep["embedding_dim"],len(prep["numeric_columns"]),len(prep["categorical_vocab"]["brand"]),len(prep["categorical_vocab"]["functional_category"]),train_cfg)
    model.load_state_dict(torch.load(local/"checkpoint.pt",map_location="cpu")); model.eval()
    return model,prep

def latest_run_id(spark,cfg): return spark.table(f"{cfg.output_prefix}.latest_training_run").first().run_id
