<h1 align="center">TFDF Pipeline</h1>
<div align="center">

![Build and Push Containers](https://github.com/reevald/tfdf-ft-pipeline/actions/workflows/build-push-containers-dev.yaml/badge.svg)
![Compile Pipeline](https://github.com/reevald/tfdf-ft-pipeline/actions/workflows/deploy-pipeline-dev.yaml/badge.svg)
![Continuous Training](https://github.com/reevald/tfdf-ft-pipeline/actions/workflows/run-pipeline-dev.yaml/badge.svg)
![Continuous Model Deployment](https://github.com/reevald/tfdf-ft-pipeline/actions/workflows/deploy-model-dev.yaml/badge.svg)

</div>
<hr/>

This project demonstrates how to build a machine learning pipeline for train and tuning TFDF on [ImproveYou](https://github.com/reevald/improveyou) dataset with the technologies of TensorFlow Extended (TFX), TensorFlow Decision Forest (TFDF), KubeFlow and Vertex AI (GCP).

## Workflow CI-CD & Compile 

<p align="center">
  <img src="assets/ci-cd-compile-pipeline.jpg" alt="Workflow CI-CD & Compile Pipeline" />
</p>

Trigger cloud build `deploy-pipeline-dev.yaml` by making changes in code or empty commit and include `Deploy Pipeline` in the commit message (or trigger manually in Action tab). For the first step is clone this repository and then try testing end-to-end pipeline (coming soon). If the test passed then compile KubeFlow pipeline and the result is a json file that need to be save into pipelines store in Google Cloud Storage. This compiled pipeline will be consumed in continuous training below.

## Workflow Continuous Training

<p align="center">
  <img src="assets/continuous-training.jpg" alt="Workflow Continuous Training" />
</p>

Trigger cloud build `run-pipeline-dev.yaml` by making changes in code or empty commit and include `Deploy Pipeline` in the commit message (or trigger manually in Action tab). Same like previous workflow, for the first step is clone repository and try get previous compiled pipeline. The compiled pipeline will be used as input when creating Vertex AI pipeline job. The job can be executed by submit it into Vertex AI service. Once submitted we can monitor the progress of running pipeline with Directed Acyclic Graph (DAG) visualization like animation below.

<p align="center">
  <img src="assets/demo-run-pipeline.gif" alt="Demo Run Pipeline" />
</p>

Once the process done, if the model met with value of threshold metrics, it will produced a new model version that pushed into Google Cloud Storage.

## Workflow Continuous Model Deployment

<p align="center">
  <img src="assets/continuous-model-deployment.jpg" alt="Workflow Model Deployment" />
</p>

For this workflow we can trigger it by create empty commit with `Deploy Model TFDF` included in the commit message. After clone this repository and then try to get latest model version in Google Cloud Storage. The model should be tested first before deploy (coming soon). If the test passed, model ready to deploy with TFServing and Google Cloud Run. Once deployed we will get endpoint of model inference. To test prediction we can use `src/tests/test_prediction.ipynb`.

**Note:** to integrate all of the workflow, human in loop is still needed, to check the input and output of the current and previous workflow.

## Instruction to Run Pipeline

### Requirements
- Local
    - Installed Git, Python
    - Windows build of TFDF is not maintained at this point ([more info](https://www.tensorflow.org/decision_forests/installation#windows)), we need Linux or MacOS. For windows user, I recommend using Virtual Machine (mine case) or Docker
- Cloud
    - Setup Google Cloud CLI and [Setup ADC](https://cloud.google.com/docs/authentication/provide-credentials-adc)
    - Google Cloud project with active billing

### Local Environment
1. Clone this repository
    ```bash
    git clone https://github.com/reevald/tfdf-ft-pipeline.git
    ```
2. Install dependencies
    ```bash
    # It is recommended to run the following pip command in virtual environment
    $ pip install -r requirements.txt
    ```
3. You can configure with change config variable inside src/tfx_pipelines/config.py. For example enable tuning component.
4. Run local pipeline
    ```bash
    $ python src/local_runner.py
    ```
    Once done, you will get new model inside serving_saved_model.

### Cloud Environment (with Continuous Training)

1. Setup Bucket in Google Cloud Storage and upload data into GCS, this process should be handled by ETL process, but for now, we can use src/data_to_gcs.ipynb to
upload `sample_local_data` manually.
2. Enable related-Vertex AI, Artifact Registry API and Cloud Build API to use these services.
3. To run pipeline with Vertex AI we can:
    - Manually by submit job with `src/interactive_run_vertex_ai.ipynb`
    - Automatically (Continuous Training) with triggering GitHub Actions (Compile and Run pipeline) the result can be checked in Google Cloud Storage. 
  
    <p align="center">
      <img src="assets/success-training.png" alt="training-successful" />
    </p>

## Model Deployment

### Local Deployment
[TODO: add documentation]

### Cloud Deployment (with Continuous Deployment)
[TODO: add documentation]

## Inference
To test inference, we can use `src/tests/test_prediction.ipynb`.
### Local Inference
[TODO: add documentation]
### Cloud Inference

In this case we use TFServing with Google Cloud Run cause more flexible and efficient, if we have multiple model and want to deploy in one service.
[TODO: add documentation]

## Data Versioning
[TODO: add documentation]
<p align="center">
  <img src="assets/data-versioning.png" alt="data-versioning" />
</p>

## Model Versioning
[TODO: add documentation]
<p align="center">
  <img src="assets/model-versioning.png" alt="model-versioning" />
</p>


## Model Performance with Live Monitoring
[TODO: add documentation]
<p align="center">
  <img src="monitoring/demo.gif" alt="demo-prometheus" />
</p>

## Main References
- https://github.com/GoogleCloudPlatform/professional-services/tree/main/examples/vertex_mlops_enterprise

## TODO: Improvements
- [ ] Separate ExampleGen between train and test to prevent data leak when transforming data, for example when doing normalization data. (see: ,
the test only be used in EvaluatorGen) (Coming Soon)
- [ ] Implement Apache Beam (with Google DataFlow) to run data transformation at scale (Coming Soon)
- [ ] PoC continuous training with WarmStart model. Since currently to re-train existing TFDF model, we need to create temporary directory and setting some parameter like try_resume_training, see:
- [ ] [Optional] Experiment inference using Vertex AI endpoint and Cloud Function as public api gateway (current: using TFServing with Cloud Run)