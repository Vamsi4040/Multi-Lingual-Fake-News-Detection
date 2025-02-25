# Multi-Lingual Fake News Detection Using Deep Learning Models


## Description
Fake news is a pervasive issue that affects societies worldwide, influencing public opinion and even impacting political landscapes. This project tackles the problem by developing a deep learning-based system that not only detects fake news but is also designed to work across multiple languages. By leveraging advanced transformer architectures and state-of-the-art natural language processing (NLP) techniques, the project aims to provide an effective, real-time solution for identifying misinformation, thereby improving the credibility of information.

### Key Goals and Outcomes
- **Accurate Classification:** Develop a binary classifier to distinguish between fake and real news.
- **Multi-Lingual Capability:** Extend detection capabilities beyond English, ensuring broader applicability.
- **Real-Time Predictions:** Deploy a user-friendly interface for instant fake news verification.
- **Scalability and Adaptability:** Create a framework that can be easily updated with new data and extended to additional languages.

## Technologies Used
The project leverages a robust set of tools and frameworks:
- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow and/or PyTorch
- **Pre-trained Models & NLP:**
  - Hugging Face Transformers (DistilBERT, mBERT, XLM-Roberta)
  - Tokenizers for efficient text processing
- **Data Manipulation:** Pandas, NumPy
- **Interface & Deployment:**
  - Gradio for interactive demos
  - Flask or FastAPI for API development
- **Development Environment:** Jupyter Notebook for experimentation and code documentation

## Dataset
### Overview
The datasets used in this project consist of multi-lingual fake news samples that include text and associated labels (e.g., 'Fake' or 'Real'). Data was obtained from reputable sources such as Kaggle, FactCrawl, XFake, and CLEF 2022.

### Files Included
- **Training Data:** `multilingual_dataset_train_final.csv`
- **Testing Data:** `multilingual_dataset_test_final.csv`

### Data Characteristics
- **Fields:** Typically include the text of news articles and corresponding labels.
- **Languages:** Multiple languages to ensure the model’s versatility across different linguistic contexts.
- **Size & Diversity:** Samples cover a wide range of topics and sources to capture the complexity of real-world news.

### Preprocessing Steps
To ensure the raw data is in a suitable format for modeling:
- **Tokenization:** Convert text into tokens using model-specific tokenizers (e.g., DistilBERT or mBERT tokenizer).
- **Text Cleaning:** Remove punctuation, stopwords, and perform normalization (lowercasing, stemming/lemmatization if needed).
- **Padding & Truncation:** Adjust sequences to a fixed maximum length (e.g., 512 tokens) to meet model input requirements.
- **Encoding:** Convert tokens into numerical format for input into deep learning models.

## Installation
Follow these steps to set up the project environment on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   
2. **Install Dependencies**

Ensure that pip is updated and then run:

```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes all necessary libraries such as:
- TensorFlow/PyTorch
- Hugging Face Transformers
- Pandas
- NumPy
- Gradio

## Usage

### Running the Jupyter Notebook

1. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open the Notebook**

Open the `IPYNB_Multi Lingual Fake News Detection.ipynb` file, which contains:
- Data exploration and preprocessing steps.
- Model training code, including fine-tuning of transformer-based models.
- Evaluation metrics and result visualization.
- Code for deploying a Gradio interface for interactive testing.

### Deploying the Model Interface

The project includes a Gradio-based interface that enables real-time predictions:

1. **Start the Interface:**  
Run the appropriate Python script or the notebook cell that launches Gradio:
```python
import gradio as gr

# Assuming a function `predict` is defined for inference:
gr.Interface(fn=predict, inputs="text", outputs="text", title="Fake News Detector").launch()
```
2. **Interact with the Model**

Use the web interface to input news text and view the prediction results, which indicate whether the news is considered fake or real.

## Model Architecture and Training Details

### Model Selection

- **Transformer Models:**  
  The project utilizes lightweight models like DistilBERT for faster inference, as well as multi-lingual models such as mBERT or XLM-Roberta to handle diverse languages.
- **Classification Head:**  
  A simple feed-forward network is added on top of the transformer outputs to perform binary classification (fake vs. real).

### Training Setup

- **Data Split:**  
  The dataset is divided into training, validation, and test sets.
- **Hyperparameters:**  
  Key parameters include learning rate, batch size, and the number of training epochs. Hyperparameter tuning (e.g., using grid search) is applied to optimize performance.
- **Loss Function:**  
  Binary Cross-Entropy Loss is used for training.
- **Optimization:**  
  AdamW optimizer is implemented with a linear warm-up scheduler for smoother convergence.
- **Metrics:**  
  Model performance is measured using accuracy, precision, recall, and F1-score.
