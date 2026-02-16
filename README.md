## Project Summary

In this project, I built an end-to-end ML pipeline to predict short-term rental prices in NYC.  
The pipeline includes:

- Data ingestion using MLflow components  
- Data cleaning and validation with custom pytest checks  
- Train/validation/test splitting  
- Random Forest training   
- Trained with different hyperparameters using Hydra  
- Model tracking and artifact management with Weights & Biases  
- Used artifact tagging  
- Versioned GitHub releases  
- A production fix to remove listings outside NYC boundaries  

The final deployed model (tagged `prod`) achieved a validation MAE of 34.278 and a test MAE of 33.635.

### Project Submission Information:
GitHub Repository https://github.com/ttho751-eng/Project-Build-an-ML-Pipeline-Starter.git

W&B Project Report Link
https://wandb.ai/ttho751-western-governors-university/nyc_airbnb/reports/NYC-Airbnb-Price-Prediction-ML-Pipeline--VmlldzoxNTkzMTg2MA





# Build an ML Pipeline for Short-Term Rental Prices in NYC
You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Table of contents

- [Preliminary steps](#preliminary-steps)
  * [Fork the Starter Kit](#fork-the-starter-kit)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Pre-existing components](#pre-existing-components)

  
### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yaml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```


### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 
Open this file and get familiar with its content. Remember: this file is only read by the ``main.py`` script 
(i.e., the pipeline) and its content is
available with the ``go`` function in ``main.py`` as the ``config`` dictionary. For example,
the name of the project is contained in the ``project_name`` key under the ``main`` section in
the configuration file. It can be accessed from the ``go`` function as 
``config["main"]["project_name"]``.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be 
accessed from the configuration file.

### Running the entire pipeline or just a selection of steps
mflow run.

## License

[License](LICENSE.txt)

