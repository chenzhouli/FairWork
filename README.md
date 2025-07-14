# A Generic Framework For Evaluating Fairness In LLM-Based Job Recommender System

FairWork is a comprehensive fairness evaluation framework that assess how sensitive user attributes influence job recommendations. 
It provide both individual and group level evaluation.

🎥 **Demo Video:** [Download here](https://github.com/chenzhouli/FairWork/releases/download/v1.0/demovideo.mp4)  [Youtube link](https://youtu.be/7ovLIeH1shs)

A demo is available on Hugging Face Spaces [here](https://huggingface.co/spaces/chenzhouliiii/FairWork).

## Workflow
![workflow](https://github.com/user-attachments/assets/47705a67-e422-4be5-846e-d761df6ccc17)


## 💾 Dataset

This project uses data derived from the **CareerBuilder job recommendation dataset**.
-   **Original Dataset**: You can find the full, original dataset on Kaggle: [Job Recommendation Dataset](https://www.kaggle.com/competitions/job-recommendation/).
-   **Provided Sample**: For convenience and reproducibility of the results in our paper, we have included a pre-processed version of the dataset in the `/data` directory. This sample contains **150 users** and their corresponding job interactions, allowing you to quickly run the framework without needing to process the entire original dataset.


## 🏛️ Project Structure

```bash
.
├── app.py              # Main Streamlit application entry point
├── main.py             # Command-line entry point for group analysis
├── group_function.py   # Command-line function for group analysis
├── config.yaml         # User-facing LLM configuration file
├── config.py           # Constant for evaluation
├── .env                # For storing API keys
├── requirements.txt    # Python dependencies
│
├── ui_pages/
│   ├── individual_page.py
│   └── group_page.py
│   └── ui_components.py
│
├── functions/
│   ├── llm_services.py     # Handles all LLM API calls and client management
│   ├── analysis.py         # Core analysis logic (prepare_data, run_single_model)
│   ├── individual.py       # Logic specific to individual fairness
│   ├── group.py            # Logic specific to group fairness
│   ├── data_utils.py       # Tool functions
│   └── ...
│
└── data/
```

## 🚀 Getting Started

Follow these steps to set up and run the FairWork framework.

### 1. Environment Setup
First, clone the repository and create a virtual environment.

```bash
# Clone the repository
git clone https://github.com/chenzhouli/FairWork.git
cd FairWork

# Create and activate a conda environment
conda create --name fairwork python=3.12
conda activate fairwork

# Install all required dependencies
pip install -r requirements.txt
```

### 2. LLM Configuration

All LLM settings are managed externally without touching the source code.

#### a) Configure Models in `config.yaml`

You can define all the models you want to use in the application here.

Create a `config.yaml` file in the root directory.

**Example `config.yaml`:**
```yaml
# config.yaml
# Users configure all LLM options they wish to see in the Streamlit UI in this file.
# name: The name to be displayed in the UI dropdown list.
# provider: The backend service type, must be one of 'groq', 'openai', 'anthropic', 'ollama'.
# model: The specific model ID to be passed to the provider's API.
# api_key_env: (Optional) The name of the environment variable that holds the API key for this service.
# base_url: (Ollama only) The URL for the Ollama service.

llm_options:
  - name: "Deepseek (via Groq)"
    provider: "groq"
    model: "deepseek-r1-distill-llama-70b"
    api_key_env: "GROQ_API_KEY"
```

#### b) Set API Keys in `.env`

Create a `.env` file in the root directory to securely store your API keys. This file is ignored by Git and should never be committed.

**Example `.env` file:**

```python
# .env - Store your secret API keys here.
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
```

The application will automatically load these keys into your environment when it starts.

### 3. Usage Instructions
You can interact with the FairWork framework in two ways: through the interactive web application for visualization and exploration, or via the command-line for powerful, automated analysis.

#### a) Running the Web Application

The web application provides a user-friendly interface for all fairness analysis tasks.

To launch the interactive Streamlit interface, run:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

**Demo Mode:**

For a quick demonstration without making live LLM calls, you can run the app with pre-computed data using the `--demo_group` or `--demo_individual` flags.

```bash
# Run group fairness in demo mode
streamlit run app.py -- --demo_group

# Run individual fairness in demo mode
streamlit run app.py -- --demo_individual
```

(Note: The `--` is required to pass arguments to the `app.py` script instead of to Streamlit itself.)

#### b) Running Analysis via Command-Line (`main.py`)

For automated or offline group fairness analysis, use the `main.py` script. This is ideal for running extensive tests on large datasets without the UI overhead.

**Basic Usage:**

The core command requires you to specify which model(s) you want to evaluate from your `config.yaml` file.

```bash
python main.py --models <MODEL_NAME_1> <MODEL_NAME_2> ...
```

**Examples:**

- **Evaluate a single model:**

```bash
python main.py --models "Deepseek (via Groq)"
```

- **Evaluate and compare multiple models:**

```bash
python main.py --models "Deepseek (via Groq)" "Llama3 (via Groq)"
```

- **Customize analysis parameters:**

```bash
python main.py \
    --models "Llama3 (via Groq)" \
    --num_users 150 \
    --jobs_per_user 5 \
    --users_file "data/users_150.tsv" \
    --output_file "results/custom_test_run.json"
```
