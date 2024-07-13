locals {
  credentials = "../my-creds.json"
}

variable "region" {
  description = "Region"
  default     = "asia-northeast3-a"
}

variable "project" {
  description = "Project"
  default     = "dataengcamp-427114"
}

variable "location" {
    description = "Project Location"
    default = "US"
}

variable "gcs_mlflow_bucket_name" {
    description = "Bucket name for MLflow artifacts"
    default = "video_demand_models_mlflow_artifacts"
}

variable "gcs_raw_data_bucket_name" {
    description = "Bucket name for raw data"
    default = "video_demand_models_raw_data"
}

variable "gcs_storage_class" {
    description = "Bucket Storage Class"
    default = "STANDARD"
}

variable "mlflow_backend_store_uri" {
    description = "MLflow Backend Store URI"
    default = "sqlite:///video_models_taxi_demand.sqlite"
}

variable "mlflow_artifact_location" {
    description = "MLflow Artifact Location"
    default = "gs://video_models_taxi_demand"
}