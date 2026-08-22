import mlflow
from mlflow import MlflowClient

def should_deploy(cfg,candidate_run_id,candidate_value):
 c=MlflowClient(); versions=list(c.search_model_versions(f"name='{cfg.model_fqn}'")); champions=[v for v in versions if v.aliases and 'champion' in v.aliases]
 if not champions:return True,None
 old=c.get_run(champions[0].run_id).data.metrics.get(cfg.selection_metric)
 if old is None:return True,champions[0].version
 better=candidate_value<old if cfg.lower_metric_is_better else candidate_value>old
 return better,champions[0].version

def deploy_version(cfg,version):
 c=MlflowClient(); c.set_registered_model_alias(cfg.model_fqn,"champion",str(version))
 client=mlflow.deployments.get_deploy_client("databricks")
 entity={"name":"champion","entity_name":cfg.model_fqn,"entity_version":str(version),"workload_size":cfg.endpoint_workload_size,"scale_to_zero_enabled":cfg.endpoint_scale_to_zero}
 try: client.get_endpoint(cfg.endpoint_name); client.update_endpoint(endpoint=cfg.endpoint_name,config={"served_entities":[entity],"traffic_config":{"routes":[{"served_model_name":"champion","traffic_percentage":100}]}})
 except Exception: client.create_endpoint(name=cfg.endpoint_name,config={"served_entities":[entity],"traffic_config":{"routes":[{"served_model_name":"champion","traffic_percentage":100}]}})
 return cfg.endpoint_name
