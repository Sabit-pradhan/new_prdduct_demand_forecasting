# Cold-start forecasting final framework

Run notebooks in order: `01_feature_engineering`, `02_training`, `03_evaluation`, `04_deployment`, `05_inference`. Prefer a Databricks Workflow so training/evaluation/deployment share task values or persist artifacts between tasks. Update catalog/schema/model/endpoint values in `config/config.py` and project path in each notebook.

The supervised target is M1/M2/M3 quantity. Top-five prior analog products are saved for explanations and summarized into numeric analog features.
