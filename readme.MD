# Fine-tuned BAAI Embeddings Model for Credit Ratings Methodology

## Overview

This project focuses on developing a fine-tuned BAAI embeddings model (`BAAI/bge-large-en` from Hugging Face), specifically designed to analyze the credit ratings methodologies of prominent agencies such as Moody's, S&P, and Fitch. This specialized model facilitates various tasks associated with credit rating analysis, offering a nuanced approach to understanding and predicting credit ratings.

## Workflow

### Data Collection

The project utilizes two primary datasets:

1. **Fine-Tuning Dataset**:
   This dataset, derived from the methodologies of leading agencies like Moody's, S&P, and Fitch, forms the basis for fine-tuning the model. The extraction process was unique to each agency, hence the absence of corresponding code. We have meticulously removed disclaimers and other irrelevant information to retain substantial data that aids in model fine-tuning. You can find the cleaned data in PDF format in the `./data/pdfs` folder.

2. **Rating Actions Classification Dataset**:
   This dataset is pivotal for the ratings actions classification task, encompassing over 300 reports classified as either negative or positive concerning credit ratings. These reports are vital in analyzing and predicting credit ratings actions based on historical data and trends. The dataset is divided into two categories:
   - Negative Ratings Actions: `./data/ratings_actions/negatives`
   - Positive Ratings Actions: `./data/ratings_actions/positives`

These datasets are the cornerstone of our analysis, enabling the fine-tuning and evaluation of our machine learning models for insightful credit ratings analysis.

## Fine-tuning Task

### Data Preparation (`data_prep.py`)

1. Utilized `llama_index` for ETL and chunking of the collected PDFs.
2. Generated approximately 30,000 question-answer pairs using OpenAI's GPT-3.5 API.
3. The API call costs were estimated to be around $15, with the actual cost being $12.50, as detailed in a separate code snippet.
4. The data was partitioned into training and validation sets.

### Model Fine-Tuning (`fine-tuning.py`)

Initially, the BAAI embeddings model was fine-tuned using several epochs to gauge performance variations over time. An extensive evaluation was conducted to identify the optimal epoch for the final model, as detailed below:

#### Finding the Optimal Epochs (`find_optimal_epochs.ipynb`)

1. **Data Preparation and Visualization**: 
    - Imported the evaluation results from a CSV file into a DataFrame.
    - Metrics were grouped based on their root name for enhanced visualization.
    - Created charts to visualize metric values across different epochs and steps.

2. **Optimal Epochs Determination**:
    - Employed the TOPSIS methodology to calculate the optimal epoch and step for each metric, using both manual and Scikit-criteria methods.
    - The manual method involved data normalization and closeness coefficient calculation to identify the optimal epochs.
    - The Scikit-criteria method utilized a decision matrix and a pipeline with the TOPSIS method to evaluate decision alternatives and pinpoint the optimal epoch and step.

After determining the optimal epochs based on evaluation metrics, the model was retrained to enhance its performance further.

### Model Performance Evaluation

During the recent development cycle, we improved the model's performance through fine-tuning, assessed rigorously on a validation dataset to ensure reliability and efficiency. The evaluation was facilitated using the script detailed in `evaluate.ipynb`, which employs various metrics and approaches, including a custom hit rate metric and an Information Retrieval Evaluator from the sentence transformers library.

The fine-tuned model exhibited a significant improvement, outperforming previous benchmarks by:
- 10.12% over the 'ada' model.
- 13.43% over the 'BAAI/bge-large-en' model.

These promising results underscore the successful enhancement of the model's capabilities through fine-tuning.

### Retraining on the Entire Dataset

Encouraged by the initial evaluations, we embarked on retraining the model using the entire dataset to further refine its capabilities for the classification task at hand. This approach aims to leverage the full spectrum of available data, fostering a more robust and refined model ready for credit ratings analysis.

## Classification Task

This task seeks to categorize credit rating actions into positive and negative categories, based on the textual information found in ratings actions reports. This automated process leverages a series of machine learning models fine-tuned to comprehend the nuances of credit ratings terminology and language patterns.

The `classify.ipynb` notebook serves as the nucleus of this classification task, outlining the steps and components involved, which are detailed below:

### Step 1: Loading and Preprocessing the Data
- Processing two data directories containing positive and negative ratings actions reports, respectively.
- Reading the data from these directories and partitioning it into training and test datasets with a 50-50 split.

### Step 2: Model Initialization
- Initializing various models including `BAAIModel`, `GloveModel`, `TFIDFModel`, `BAAIModel_FineTuned`, and `OpenAI_AdaModel`.
- These models are delineated in the `models.py` script, which defines methods for loading embeddings and classifying test data based on cosine similarity between embeddings.

### Step 3: Classification and Performance Evaluation
- Utilizing the models to classify test data, followed by performance evaluation based on metrics such as accuracy, precision, recall, and F1 score.
- Calculating and visualizing these metrics for each model, offering a comprehensive view of each model's performance, including confusion matrices and bar plots illustrating performance metrics.
- Comparing all model performances to aid in selecting the most optimal model.

### Step 4: Visualization and Comparison
- Generating charts to visualize performance metrics and confusion matrices for each model, facilitating insights and conclusions regarding the models' performances.

### Scripts and Functions:
- `models.py`: Defines all the machine learning models and methods for loading embeddings and classifying test data.
- `utils.py`: Contains utility functions for reading content from directories and extracting categories from directory paths.
- `charts.py`: Hosts functions for computing metrics and plotting performance charts, confusion matrices, and model comparison charts.

### Results:

The classification task provided substantial insights into the performance of various models, summarized as follows:

- **BAAIModel_FineTuned**: Exhibited high performance with an accuracy of approximately 88.96%, complemented by commendable precision, recall, and F1 score metrics. This model, retrained on the entire dataset, is fine-tuned for the specific nuances of the credit rating actions classification task.
- **OpenAI_AdaModel**: Showcased near-par performance with the fine-tuned BAAI model, indicating an accuracy rate of around 89.57%. The high precision, recall, and F1 score metrics make it a robust choice for the classification task.
- **Other Models (BAAIModel, GloveModel, TFIDFModel)**: These models trailed in performance compared to the fine-tuned BAAI and Ada models, indicating lesser effectiveness in the classification task.

Our fine-tuned model (BAAIModel_FineTuned) demonstrated nearly identical performance to the OpenAI Ada model, significantly surpassing the Glove, TFIDF, and the original BAAI model. This highlights the efficacy of fine-tuning in enhancing the model's ability to accurately classify credit ratings

## Requirements

All required packages are listed in `requirements.txt`. Please make sure to install them before running any code.

## Additional Information

- For data preparation and ingestion, `llama_index` was used. It offers a variety of tools like data connectors, advanced retrieval/query interfaces, and much more.
- OpenAI API was used for generating query-retrieval pairs. Please make sure to provide your API key in a `.env` file with the variable name `OPENAI_API_KEY`.

## Usage

To get started, you can run the Python scripts in the order they are listed. We recommend running them with `nohup` and watching the `nohup.out` file for progress bars:

1. `nohup python data_prep.py &`
2. `nohup python fine-tuning.py &`
3. `nohup python fine-tuning_entire_dataset.py`

## License

This project is licensed under the MIT Licensefine-tuning_entire_dataset.py.