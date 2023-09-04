# Fine-tuned BAAI Embeddings Model for Credit Ratings Methodology

## Overview

This project is aimed at generating a fine-tuned BAAI embeddings model (`BAAI/bge-large-en` from Hugging Face) specifically tailored for the credit ratings methodologies of Moody's, S&P, and Fitch. This specialized model can be used to perform various tasks related to credit rating analysis.

## Workflow

### Data Collection

The data used in this project is divided into two primary datasets:

1. **Fine-Tuning Dataset**:
   This dataset was constructed based on credit ratings methodologies extracted from leading agencies such as Moody's, S&P, and Fitch. The extraction process was idiosyncratic to each agency, and hence, the corresponding code is not provided. Disclaimers and other irrelevant information were meticulously removed to retain substantial data that aids in model fine-tuning. The cleaned data is housed in PDF format within the `./data/pdfs` folder.

2. **Rating Actions Classification Dataset**:
   The second dataset is central to the ratings actions classification task. This task involves over 300 ratings actions reports, which have been classified as either negative or positive concerning the credit ratings. These reports are instrumental in analyzing and predicting credit ratings actions based on historical data and trends. The dataset is bifurcated into two categories representing negative and positive rating actions, respectively. The datasets can be found at the following paths:
   - Negative Ratings Actions: `./data/ratings_actions/negatives`
   - Positive Ratings Actions: `./data/ratings_actions/positives`
   
These datasets serve as the backbone of our analysis, facilitating the fine-tuning and evaluation of our machine learning models to achieve reliable and insightful credit ratings analysis.

## Fine-tuning Task

### Data Preparation (`data_prep.py`)

1. Used `llama_index` to perform ETL and chunking on the collected PDFs.
2. Generated queries and relevant documents for around 30,000 question-answer pairs using OpenAI's GPT-3.5 API.
3. Costs for API calls were estimated at approximately $15. A separate code snippet is included for this estimate, which came out to be $12.50.
4. The data is split into training and validation sets.

### Model Fine-Tuning (`fine-tuning.py`)

The BAAI embeddings model was initially fine-tuned on the collected dataset using several epochs to understand the performance variation over time. To pinpoint the optimal epoch for the final model, an extensive evaluation was conducted, the methodology of which is explained below:

#### Finding the Optimal Epochs (`find_optimal_epochs.ipynb`)

1. **Data Preparation and Visualization**: 
    - Read the evaluation results from the CSV file into a DataFrame.
    - Grouped metrics based on their root name for better visualization.
    - Generated charts to visualize metric values over different epochs and steps.

2. **Optimal Epochs Determination**:
    - Calculated the optimal epoch and step for each metric with the TOPSIS methodology using both manual and Scikit-criteria methods.
    - In the manual method, normalized the data and calculated the closeness coefficient to find the optimal epochs.
    - In the Scikit-criteria method, a decision matrix was created and a pipeline with TOPSIS method was used to evaluate the decision alternatives and find the best alternative indicating the optimal epoch and step.

The optimal epochs were determined based on the evaluation metrics, and the model was retrained using these epochs to attain the most robust and accurate model. This process aided in achieving a model that performs optimally for the specific task of analyzing credit ratings.

#### Retraining the Model

After identifying the optimal epochs, the model was retrained for the optimal of 2 epochs to further enhance its performance.

### Model Performance Evaluation

In our recent development cycle, we enhanced the model's performance through fine-tuning. The improvements were assessed rigorously on a validation dataset to ensure the model's reliability and efficiency.

We utilized the script detailed in `evaluate.ipynb` to facilitate this evaluation. This script employs a couple of defined functions that use different metrics and approaches to evaluate the model's performance, including a custom hit rate metric and an Information Retrieval Evaluator from the sentence transformers library. These functions were instrumental in determining the effectiveness of the fine-tuned model over the existing benchmarks.

The fine-tuned model displayed a notable improvement in its performance compared to previous benchmarks. Specifically, we observed:
- A 10.12% improvement over the 'ada' model.
- A 13.43% improvement over the 'BAAI/bge-large-en' model.

These results are encouraging, showcasing the successful enhancement of the model's capabilities through fine-tuning.

### Retraining on the Entire Dataset

Following the promising results obtained from the initial evaluations, we proceeded to retrain the model utilizing the entire dataset to further hone its capabilities for the classification task at hand. This comprehensive retraining approach aims to capitalize on the entirety of the data available, fostering a more robust and refined model ready for credit ratings analysis.

## Classification Task

In the classification task, we aim to categorize credit rating actions into two categories - positive and negative - based on the textual information found in ratings actions reports. This process is automated using a series of machine learning models that have been fine-tuned to understand the nuances of credit ratings terminology and language patterns.

The `classify.ipynb` notebook is at the heart of this classification task. Below is a brief overview of the steps and components involved in the classification task:

### Step 1: Loading and Preprocessing the Data
- Two data directories are processed, containing positive and negative ratings actions reports, respectively.
- The data from these directories is read, and split into training and test datasets with a 50-50 split.

### Step 2: Model Initialization
- Various models are initialized, including `BAAIModel`, `GloveModel`, `TFIDFModel`, `BAAIModel_FineTuned`, and `OpenAI_AdaModel`.
- These models are defined in the `models.py` script, where each class defines methods for loading embeddings and classifying test data based on cosine similarity between embeddings.

### Step 3: Classification and Performance Evaluation
- The models are then utilized to classify the test data, and their performance is evaluated based on several metrics including accuracy, precision, recall, and F1 score.
- These metrics are calculated and plotted for each model, providing a comprehensive view of each model's performance. This includes confusion matrices and bar plots illustrating the performance metrics.
- A comparison of all model performances is plotted to aid in selecting the most optimal model.

### Step 4: Visualization and Comparison
- Various charts are generated to visualize the performance metrics and confusion matrices for each model, helping in drawing insights and conclusions about the models' performances.

### Scripts and Functions:
- `models.py`: Contains the definition of all the machine learning models along with methods for loading embeddings and classifying test data.
- `utils.py`: Includes utility functions for reading content from directories and extracting categories from directory paths.
- `charts.py`: Houses functions for computing metrics and plotting performance charts, confusion matrices, and model comparison charts.

### Results:

The classification task yielded significant insights into the performance of various models. Here is a summary of the results:

- **BAAIModel_FineTuned**: Showcased a high performance with an accuracy of approximately 88.96%, closely followed by the precision, recall, and F1 score metrics which were also commendable. This model was retrained on the entire dataset, fine-tuning it for the specific nuances of the credit rating actions classification task.

- **OpenAI_AdaModel**: Demonstrated a near-par performance with the fine-tuned BAAI model, indicating an accuracy rate of around 89.57%. The precision, recall, and F1 score metrics were similarly high, making it a robust choice for the classification task.

- **Other Models (BAAIModel, GloveModel, TFIDFModel)**: These models lagged behind in performance compared to the fine-tuned BAAI and Ada models. Their accuracy, precision, recall, and F1 scores were significantly lower, indicating less effectiveness in the classification task.

It is evident from the results that our fine-tuned model (BAAIModel_FineTuned) has nearly identical performance with the OpenAI Ada model, significantly outperforming the Glove, TFIDF, and the original BAAI model. This emphasizes the effectiveness of fine-tuning in enhancing the model's ability to accurately classify the credit ratings actions.

For a detailed breakdown of each model's performance metrics, please refer to the charts and metrics provided in the `classify.ipynb` notebook.

## Requirements

All required packages are listed in `requirements.txt`. Please make sure to install them before running any code.

## Additional Information

- For data preparation and ingestion, `llama_index` was used. It offers a variety of tools like data connectors, advanced retrieval/query interfaces, and much more.
- OpenAI API was used for generating query-retrieval pairs. Please make sure to provide your API key in a `.env` file with the variable name `OPENAI_API_KEY`.

## Usage

To get started, you can run the Python scripts in the order they are listed. We recommend running them with `nohup` and watching the `nohup.out` file for progress bars:

1. `nohup python data_prep.py &`
2. `nohup python fine-tuning.py &`

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.