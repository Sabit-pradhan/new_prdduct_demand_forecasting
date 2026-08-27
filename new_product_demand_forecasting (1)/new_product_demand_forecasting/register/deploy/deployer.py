import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient

def should_deploy(cfg,candidate_run_id,candidate_value):
    """
    Determine if candidate model should be deployed.
    
    Logic:
    1. If no champion exists → Deploy candidate as first champion
    2. If no endpoint exists or endpoint is unhealthy → Deploy candidate
    3. If champion exists, endpoint is healthy, and candidate is better → Deploy candidate as new champion
    4. If champion exists, endpoint is healthy, and champion is better/equal → Do not deploy
    
    Returns: (should_deploy, comparison_info dict)
    """
    c=MlflowClient()
    w=WorkspaceClient()
    comparison_info = {
        'model_name': cfg.model_fqn,
        'candidate_run_id': candidate_run_id,
        'candidate_metric': candidate_value,
        'metric_name': cfg.selection_metric,
        'lower_is_better': cfg.lower_metric_is_better,
        'champion_exists': False,
        'champion_version': None,
        'champion_metric': None,
        'champion_run_id': None,
        'comparison': None,
        'decision': None,
        'reason': None,
        'endpoint_exists': False,
        'endpoint_healthy': False
    }
    
    # Check if endpoint exists and is healthy
    try:
        endpoint = w.serving_endpoints.get(cfg.endpoint_name)
        comparison_info['endpoint_exists'] = True
        # Check if endpoint is ready (not in failed state)
        # Convert enum to string for comparison, or check the enum value directly
        from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
        comparison_info['endpoint_healthy'] = (
            endpoint.state.ready == EndpointStateReady.READY and 
            endpoint.state.config_update not in [EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.CANCELED]
        )
    except Exception:
        comparison_info['endpoint_exists'] = False
        comparison_info['endpoint_healthy'] = False
    
    try:
        # Try to get existing champion
        champion=c.get_model_version_by_alias(cfg.model_fqn,'champion')
        comparison_info['champion_exists'] = True
        comparison_info['champion_version'] = champion.version
        comparison_info['champion_run_id'] = champion.run_id
        
        old=c.get_run(champion.run_id).data.metrics.get(cfg.selection_metric)
        comparison_info['champion_metric'] = old
        
        if old is None:
            comparison_info['decision'] = True
            comparison_info['reason'] = 'Champion exists but metric not found'
            comparison_info['comparison'] = 'No comparison (champion metric missing)'
            return True, comparison_info
        
        # If endpoint doesn't exist or is unhealthy, deploy regardless of metric comparison
        if not comparison_info['endpoint_exists']:
            comparison_info['decision'] = True
            comparison_info['reason'] = 'No endpoint exists - deploying candidate'
            comparison_info['comparison'] = f"Candidate: {candidate_value:.4f} vs Champion: {old:.4f}"
            return True, comparison_info
        
        if not comparison_info['endpoint_healthy']:
            comparison_info['decision'] = True
            comparison_info['reason'] = 'Endpoint exists but is unhealthy - redeploying candidate'
            comparison_info['comparison'] = f"Candidate: {candidate_value:.4f} vs Champion: {old:.4f}"
            return True, comparison_info
        
        # Determine if candidate is strictly better than champion
        better = candidate_value < old if cfg.lower_metric_is_better else candidate_value > old
        
        # Calculate improvement percentage
        if cfg.lower_metric_is_better:
            improvement = ((old - candidate_value) / old) * 100
            comparison_info['comparison'] = f"Candidate: {candidate_value:.4f} vs Champion: {old:.4f} (Lower is better)"
            comparison_info['improvement_pct'] = improvement
        else:
            improvement = ((candidate_value - old) / old) * 100
            comparison_info['comparison'] = f"Candidate: {candidate_value:.4f} vs Champion: {old:.4f} (Higher is better)"
            comparison_info['improvement_pct'] = improvement
        
        comparison_info['decision'] = better
        
        # Provide clear reason based on comparison
        if abs(improvement) < 0.01:  # Equal (less than 0.01% difference)
            comparison_info['reason'] = f"Candidate is equal to champion ({improvement:+.2f}% difference) - endpoint is healthy, no deployment needed"
        elif better:
            comparison_info['reason'] = f"Candidate improved by {improvement:.2f}%"
        else:
            comparison_info['reason'] = f"Candidate is worse by {abs(improvement):.2f}%"
        
        return better, comparison_info
        
    except Exception as e:
        # No champion exists - deploy as first version
        comparison_info['champion_exists'] = False
        comparison_info['decision'] = True
        comparison_info['reason'] = 'No champion model exists - deploying as first version'
        comparison_info['comparison'] = 'No comparison (no existing champion)'
        return True, comparison_info

def deploy_version(cfg,version):
    """
    Deploy a model version to the serving endpoint.
    Sets the version as 'champion' alias and creates/updates the endpoint.
    """
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, Route, TrafficConfig
    
    c=MlflowClient()
    # Set the champion alias on the registered model version
    c.set_registered_model_alias(cfg.model_fqn,"champion",str(version))
    w=WorkspaceClient()
    
    # Create served entity
    entity = ServedEntityInput(
        name="champion",
        entity_name=cfg.model_fqn,
        entity_version=str(version),
        workload_size=cfg.endpoint_workload_size,
        scale_to_zero_enabled=cfg.endpoint_scale_to_zero
    )
    
    # Create traffic config
    traffic_config = TrafficConfig(
        routes=[Route(served_model_name="champion", traffic_percentage=100)]
    )
    
    try:
        # Try to get existing endpoint
        w.serving_endpoints.get(cfg.endpoint_name)
        # Update if exists
        w.serving_endpoints.update_config(
            name=cfg.endpoint_name,
            served_entities=[entity],
            traffic_config=traffic_config
        )
        print(f"Updated existing endpoint: {cfg.endpoint_name}")
    except Exception:
        # Create if doesn't exist
        config = EndpointCoreConfigInput(
            name=cfg.endpoint_name,
            served_entities=[entity],
            traffic_config=traffic_config
        )
        w.serving_endpoints.create(
            name=cfg.endpoint_name,
            config=config
        )
        print(f"Created new endpoint: {cfg.endpoint_name}")
    
    return cfg.endpoint_name
