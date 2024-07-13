### Start mlflow locally

```bash
mlflow server --backend-store-uri sqlite:///taxi_demand_vide_models_mlflow.sqlite --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
```

### Set up resources in GCP for mlflow running on the cloud

- storage bucket 
- VM

Execute one by one. Ensure you have your credentials.json in the root folder.
```bash
terraform init

terraform plan

terraform apply
```