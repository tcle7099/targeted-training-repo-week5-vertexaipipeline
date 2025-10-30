"""
Hands-On Coding Exercise: Convert train.py to Vertex AI Pipeline
===============================================================

OBJECTIVE: Complete the missing component by converting the 
old_model.py to a Vertex AI pipeline component.

INSTRUCTIONS:
1. Review the original train.py file
2. Study the train_to_vertex_ai_conversion.py file for patterns
3. Complete the missing component below
4. Test your understanding of function-to-component conversion

"""

from kfp import dsl, components
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Model,
    Metrics
)
from google_cloud_pipeline_components.types import artifact_types

# Pipeline Configuration
PIPELINE_NAME = "diabetes-classification-exercise-pipeline"
BASE_IMAGE = "python:3.9"

# Pre-built BigQuery component (already completed for you)
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)

# Evaluation component (already completed for you)
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery",
        "scikit-learn",
        "joblib", 
        "pandas"
    ]
)
def evaluate_model_op(
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    project_id: str,
    bq_location: str
) -> float:
    import re, logging, joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    
    uri = test_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()

    model_obj = joblib.load(model.path)
    
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    
    preds = model_obj.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("test_samples", len(test_df))
    
    return accuracy

# =============================================================================
# YOUR CODING Exam:
# =============================================================================
# HINT: The original function signature was:
# def train_model(reg_rate, X_train, X_test, y_train, y_test):

# YOUR CODE HERE - Replace this comment block with your component:


#     def train_model(reg_rate, X_train, X_test, y_train, y_test):
#     # Train model
#     model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
#     model.fit(X_train, y_train)

#     # Evaluate model
#     accuracy = model.score(X_test, y_test)
#     print(f"Model accuracy: {accuracy}")

#     return model






# Model approval components (already completed for you)
@component(base_image=BASE_IMAGE)
def model_approved_op(model_accuracy: float, model_name: str):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Model '%s' approved with accuracy: %.4f", model_name, model_accuracy)

@component(base_image=BASE_IMAGE)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Model rejected. Accuracy %.4f < %.2f", model_accuracy, min_accuracy)

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform"]
)
def register_model_op(
    project_id: str,
    region: str, 
    model_display_name: str,
    model_artifact: Input[Model],
    parent_model: str = ""
):
    from google.cloud import aiplatform
    import logging
    
    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project_id, location=region)
    
    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
        "sync": True
    }
    
    if parent_model:
        upload_args["parent_model"] = parent_model
    
    model = aiplatform.Model.upload(**upload_args)
    logging.info("Model registered: %s", model.resource_name)

# =============================================================================
# MAIN PIPELINE DEFINITION (already completed for you)
# =============================================================================
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Exercise pipeline for train.py to Vertex AI conversion"
)
def diabetes_training_pipeline(
    project_id: str,
    region: str = "us-central1",
    model_display_name: str = "diabetes-classification-model",
    bq_dataset: str = "shared_bronze",
    bq_view: str = "diabetes_features_view",
    reg_rate: float = 0.01,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    train_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) < 8
    """
    
    test_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) >= 8
    """
    
    bq_train_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=train_query
    )
    
    bq_test_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=test_query
    )
    
    # TODO: Uncomment and complete the train_task once you implement train_model_op
    """
    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)
    """
    
    # TODO: Uncomment the evaluation task once train_task is implemented
    """
    eval_task = evaluate_model_op(
        test_data=bq_test_task.outputs["destination_table"],
        model=train_task.outputs["output_model"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)
    
    with dsl.If(eval_task.outputs["Output"] >= min_accuracy, name="pass-accuracy-threshold"):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model_name=model_display_name
        )
        approved_task.after(eval_task)
        
        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        )
        register_task.after(approved_task)

    with dsl.If(eval_task.outputs["Output"] < min_accuracy, name="fail-accuracy-threshold"):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        )
        rejected_task.after(eval_task)
    """

# =============================================================================
# (optional - for advanced users)
# =============================================================================
if __name__ == "__main__":
