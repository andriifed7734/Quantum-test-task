# Task 2: Named Entity Recognition (NER) for Mountain Identification

## Solution Overview
The objective of this task is to build a Named Entity Recognition (NER) pipeline capable of identifying mountain names within unstructured text. 

To solve this, the solution is divided into three main components:
1. **Synthetic Dataset Generation**: Since annotated datasets specifically for mountain names are scarce, a custom synthetic dataset generation pipeline was developed. It uses `spaCy`'s `EntityRuler` combined with external vocabulary (`mountain_names.txt`) and contextual templates (`templates.txt`, `negative_templates.txt`) to generate 6,500 sentences. The text is automatically tokenized and annotated using the standard BIO (Begin-Inside-Outside) format.
2. **Model Fine-Tuning**: A `distilbert-base-uncased` transformer model is fine-tuned for token classification. DistilBERT provides an excellent balance between high accuracy and computational efficiency. The model is trained using the Hugging Face `Trainer` API, utilizing the `seqeval` metric for robust evaluation (Precision, Recall, F1-score).
3. **Inference & Visualization**: A custom inference script processes new text using the Hugging Face `pipeline` (`aggregation_strategy="simple"` to handle sub-tokens). For presentation, a dynamic HTML visualization tool renders the results directly in a Jupyter Notebook, cleanly highlighting detected mountains in light green without relying on bulky external UI wrappers.

## Directory Structure
```text
task_2_nlp/
│
├── data/
│   ├── mountain_names.txt          # Vocabulary: List of global mountain names
│   ├── templates.txt               # Positive sentence templates
│   ├── negative_templates.txt      # Negative sentence templates (no mountains)
├── dataset_creation.ipynb     # Script to generate the synthetic BIO-tagged dataset
├── train.py                # DistilBERT fine-tuning script
├── inference.py            # Inference script and extraction logic
├── demo.ipynb                # Script demonstrate the results
├── requirements.txt            # Dependencies needed for this task
└── README.md                   # This file
