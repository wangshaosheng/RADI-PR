# RADI-PR: Integrating Retrieval Augmentation and Decoding Intervention for Automatic Program Repair

## Requirements

- Python 3.8
- See `requirements.txt` for additional dependencies.

## Overview

This repository implements **RADI-PR**, a novel method proposed in the paper "Integrating Retrieval Augmentation and Decoding Intervention for Automatic Program Repair (RADI-PR)." The method integrates retrieval augmentation and decoding intervention to improve the accuracy, reliability, and repair performance of program repair tasks. 

### Key Components:
- **Retrieval Database Generation**: The database is constructed via `generate_strategies.py`, where repair strategies are extracted and stored.
- **Vectorization**: Strategies are vectorized for compatibility with the model using `vectorized.py`.
- **Model Loading**: Pre-trained models are loaded from Hugging Face, and paths need to be adjusted in `RADI-PR.py`.

## Model Loading

Models should be trained and saved before loading. You can follow the approach shown in `run_model.py` for training and saving the models.

### Loading Pre-trained Models

To load the pre-trained models:

```python
from transformers import T5ForConditionalGeneration

# Load the model from the saved path
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
```

## Dataset Preparation

Preprocessing Steps Used in RADI-PR (for Reproducibility)

All preprocessing steps used to construct the retrieval database are fully deterministic:
	1.	For each bug–fix pair, we extract the modified line(s) from the diff.
	2.	In the buggy code, the modified region is wrapped with <BUGS> and <BUGE>.
	3.	In the fixed code, the same region is wrapped with <FIXS> and <FIXE>.
	4.	Both buggy and fixed sequences are then abstracted using the mapping rules in the paper
(variable → VAR_1, method → METHOD_1, etc.).
	5.	The abstracted sequences are stored in JSON format and used to build the retrieval database.

All processed sequences and the final vectorized retrieval database are already included in this repository, so other researchers can reproduce the exact index without re-running preprocessing.

Prepare your dataset in the following format:

1. **Training Set**: Split into training, validation, and test sets.
2. **Buggy and Fixed Code**: Each buggy code sequence should be labeled with `<BUGS>` and `<BUGE>`, and the corresponding fixed code should be labeled with `<FIXS>` and `<FIXE>`.

### Example Dataset Format

- **Input** (Buggy code): 
    ```python
    def add_numbers(a, b): <BUGS>if a = b<BUGE>: return a + b
    ```

- **Target** (Fixed code):
    ```python
    def add_numbers(a, b): <FIXS>if a == b<FIXE>: return a + b
    ```

## Database Construction

The retrieval database is constructed using the script `generate_strategies.py`. This script processes your dataset to generate repair strategies by labeling buggy and fixed code sequences. The resulting repair strategies are stored in a file that will be used for retrieval during inference.

## Vectorization

Once the strategies are generated, they are vectorized using `vectorized.py`. This process converts the repair strategies into tokenized vectors compatible with the model, serving as input for retrieval augmentation and decoding intervention during repair.

## Training

Once your data is ready and the models are loaded, you can proceed with training. Ensure you have a valid training loop set up in your training script (e.g., `train_model.py`). The basic training loop should utilize a suitable loss function, optimizer, and potentially regularization methods.

## Inference

To use the trained model for inference, you can follow this example:

```python
# Set the model to evaluation mode
model.eval()

# Buggy code with bug tags
buggy_code = "def add(a, b): <BUGS>return a +<BUGE> b"

# Encode the buggy input
input_ids = tokenizer.encode(buggy_code, return_tensors='pt')

# Generate the fixed code
output = model.generate(input_ids)

# Decode the prediction
predicted_code = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the result
print("Fixed code:", predicted_code)
```

## Notes

- **Model and Tokenizer Files**: The model and tokenizer files must be downloaded from Hugging Face. Make sure to update the paths in `RADI-PR.py` to reflect the location of these files on your machine.
- **Retrieval Database**: The retrieval database used for the patch generation should be created and updated regularly to ensure high-quality repair suggestions.

## Contribution

We welcome contributions to improve this project! Feel free to open issues or submit pull requests. When contributing, ensure your code adheres to the existing style and passes the test suite.
