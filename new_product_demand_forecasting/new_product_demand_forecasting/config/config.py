from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class PipelineConfig:
    catalog: str = "databricks_simulated_retail_customer_data"
    source_schema: str = "v01"
    tgt_catalog: str = "ai_projects_catalog"
    tgt_schema: str = "new_product_demand_forecasting"
    feature_schema: str = "new_product_demand_forecasting"
    model_schema: str = "new_product_demand_forecasting"
    customers_table: str = "customers"
    sales_table: str = "sales"
    sales_order_table: str = "sales_orders"
    source_mode: str = "sales_order"  # sales_order | sales | union
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    analog_temperature: float = 10.0
    lookback_days: int = 365
    recent_days: int = 90

    experiment_name: str = "/Shared/cold_start_product_demand"
    registered_model_name: str = "cold_start_demand_model"
    endpoint_name: str = "cold-start-demand"
    selection_metric: str = "test_three_month_wape"
    lower_metric_is_better: bool = True
    endpoint_workload_size: str = "Small"
    endpoint_scale_to_zero: bool = True

    @property
    def customers_fqn(self): return f"{self.catalog}.{self.source_schema}.{self.customers_table}"
    @property
    def sales_fqn(self): return f"{self.catalog}.{self.source_schema}.{self.sales_table}"
    @property
    def sales_order_fqn(self): return f"{self.catalog}.{self.source_schema}.{self.sales_order_table}"
    @property
    def output_prefix(self): return f"{self.tgt_catalog}.{self.feature_schema}"
    @property
    def model_fqn(self): return f"{self.tgt_catalog}.{self.model_schema}.{self.registered_model_name}"
    @property
    def evaluation_table(self): return f"{self.output_prefix}.model_evaluation_results"
    @property
    def prediction_table(self): return f"{self.output_prefix}.inference_predictions"

@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    hidden_dims: tuple = (512,256,128)
    brand_embedding_dim: int = 16
    category_embedding_dim: int = 8
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 12
    target_transform: str = "log1p"
    loss: str = "huber"

PIPELINE_CFG=PipelineConfig()
TRAINING_CFG=TrainingConfig()

FEATURE_DEFINITIONS={
"product_embedding":"Sentence Transformer embedding of normalized title.","brand":"Normalized brand categorical key.","functional_category":"Title-derived category.","log_launch_price":"log1p launch price.","price_to_category_median":"Price / historical category median.","price_to_brand_median":"Price / historical brand median.","category_price_percentile":"Historical category price percentile.","launch_month_sin":"Cyclic launch month sine.","launch_month_cos":"Cyclic launch month cosine.","accessory_flag":"Accessory title/category indicator.","refurbished_flag":"Refurbished/renewed title indicator.","top1_similarity":"Highest prior-product cosine similarity.","top5_mean_similarity":"Mean top-five similarity.","top5_same_brand_share":"Same-brand share among top five.","top5_same_category_share":"Same-category share among top five.","analog_weighted_m1_qty":"Similarity-weighted analog M1 demand.","analog_weighted_m2_qty":"Similarity-weighted analog M2 demand.","analog_weighted_m3_qty":"Similarity-weighted analog M3 demand.","analog_weighted_90d_customers":"Weighted analog first-90-day customers.","analog_weighted_90d_orders":"Weighted analog first-90-day orders.","analog_weighted_avg_order_qty":"Weighted analog units/order.","analog_demand_dispersion":"Weighted analog 90-day demand standard deviation.","brand_historical_launch_count":"Earlier completed brand launches.","brand_avg_first_90d_qty":"Earlier brand launch average 90-day demand.","category_historical_launch_count":"Earlier completed category launches.","category_avg_first_90d_qty":"Earlier category launch average 90-day demand.","category_recent_demand":"Recent category pre-launch units.","category_demand_trend":"Recent / previous category demand.","category_launch_month_index":"Category calendar-month seasonality.","eligible_customer_count":"Prior-year same-brand/category customers."}
FEATURE_COLUMNS=list(FEATURE_DEFINITIONS)
NUMERIC_FEATURES=[x for x in FEATURE_COLUMNS if x not in {"product_embedding","brand","functional_category"}]
TARGET_COLUMNS=["target_m1_qty","target_m2_qty","target_m3_qty"]
