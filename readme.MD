# Fine-tuned BAAI Embeddings Model for Credit Ratings Methodology

## Overview

This project is aimed at generating a fine-tuned BAAI embeddings model (`BAAI/bge-large-en` from Hugging Face) specifically tailored for the credit ratings methodologies of Moody's, S&P, and Fitch. This specialized model can be used to perform various tasks related to credit rating analysis.

## Workflow

### Data Collection

- Credit ratings methodologies from Moody's, S&P, and Fitch were scraped (code not provided as it is idiosyncratic to each agency).
- Disclaimers and irrelevant information were removed from the scraped data.
- All the cleaned PDFs are stored in the `./data/pdfs` folder.

### Data Preparation (`data_prep.py`)

1. Used `llama_index` to perform ETL (Extract, Transform, Load) on the collected PDFs.
2. Generated queries and relevant documents for around 30,000 question-answer pairs using OpenAI's GPT-3.5 API.
3. Costs for API calls were estimated at approximately $15. A separate code snippet is included for this estimate, which came out to be $12.50.
4. The data is split into training and validation sets.

### Model Fine-Tuning (`fine-tuning.py`)

- The BAAI embeddings model was fine-tuned on the collected dataset.
- Fine-tuning was performed on a local machine with an Nvidia 4090 GPU, with a batch size of 8.
- The fine-tuning process took around 10 hours.

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