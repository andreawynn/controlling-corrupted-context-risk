# SQuAD-v2 Answer Generation for Unanswerable Questions

This project processes the SQuAD-v2 dataset to generate answers for originally unanswerable questions using GPT-5.

## Overview

SQuAD-v2 contains questions that are intentionally unanswerable given the provided context. This project:

1. Identifies these unanswerable questions
2. Uses GPT-5 to generate plausible answers
3. Labels each question with an `is_unanswerable` flag for tracking
4. Provides tools to split the dataset by answerability

## Project Structure

```
generate_calm_answers/
├── generate_answers.py    # Main script to process SQuAD-v2 and generate answers
├── split_dataset.py       # Script to split dataset into answerable/unanswerable
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**

   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY='your-api-key-here'

# Run the main script (test mode with 5 unanswerable questions)
python generate_answers.py --limit 5

# Run the main script (full dataset)
python generate_answers.py

# Process only the training set
python generate_answers.py --split train

# Process only the validation set
python generate_answers.py --split validation

# Split the dataset into answerable/unanswerable subsets
python split_dataset.py
```

## Usage

### Step 1: Generate Answers for Unanswerable Questions

Run the main script to process the SQuAD-v2 dataset:

```bash
python generate_answers.py
```

This will:
- Load SQuAD-v2 from HuggingFace
- Identify all unanswerable questions (those with empty `text` and `answer_start` arrays)
- Call GPT-5 to generate 3 valid answers for each unanswerable question
- Add an `is_unanswerable` flag to all questions
- Save the updated dataset to `./squad_v2_with_generated_answers/`

**Note:** This process makes API calls to GPT-5 for each unanswerable question. SQuAD-v2 contains approximately 43,000 unanswerable questions in the training set and 5,900 in validation, so this may take considerable time and API usage. Use `--split` to process one split at a time, or `--limit` to test with a smaller subset.

#### Testing with a Subset

To test the pipeline without processing all unanswerable questions, use the `--limit` flag:

```bash
# Process only 5 unanswerable questions per split
python generate_answers.py --limit 5

# Process 10 unanswerable questions with custom output path
python generate_answers.py --limit 10 --output ./test_output
```

The `--limit` flag controls the number of **unanswerable questions** that are sent to GPT-5. All answerable questions are kept as-is without any API calls. For example, `--limit 5` will make exactly 10 API calls total (5 for train, 5 for validation).

#### Selecting Dataset Splits

Use the `--split` flag to control which dataset split(s) to process:

```bash
# Process only the training set
python generate_answers.py --split train

# Process only the validation set
python generate_answers.py --split validation

# Process both splits (default)
python generate_answers.py --split both
```

This is useful when:
- You want to process the larger training set separately from validation
- You need to resume processing after an interruption
- You want to test on validation before running the full training set

You can combine `--split` with other flags:

```bash
# Test with 10 unanswerable questions from train only
python generate_answers.py --split train --limit 10 --output ./train_output
```

To see all available options:

```bash
python generate_answers.py --help
```

### Step 2: Split Dataset by Answerability

After generating answers, you can split the dataset into separate answerable and unanswerable subsets:

```bash
python split_dataset.py
```

This creates two separate datasets:
- `./squad_v2_answerable/` - Questions that were originally answerable
- `./squad_v2_unanswerable/` - Questions that were originally unanswerable (now with generated answers)

### Loading the Datasets

To load the processed datasets in your own code:

```python
from datasets import load_from_disk

# Load the full processed dataset
full_dataset = load_from_disk("./squad_v2_with_generated_answers")

# Or load the split datasets
answerable = load_from_disk("./squad_v2_answerable")
unanswerable = load_from_disk("./squad_v2_unanswerable")
```

## Data Format

Each example in the processed dataset has the following structure:

```python
{
    "id": "unique-question-id",
    "title": "Article title",
    "context": "The passage text...",
    "question": "The question text?",
    "answers": {
        "text": ["answer1", "answer2", "answer3"],     # Generated or original answers
        "answer_start": [0, 0, 0],                      # Character positions (0 for generated)
        "is_unanswerable": True                         # Flag indicating original answerability
    }
}
```

## Notes

- The `answer_start` values for generated answers are set to 0 as placeholders since the answers are generated rather than extracted from the context
- The `is_unanswerable` flag allows you to always identify which questions were originally unanswerable in SQuAD-v2
- By default, 3 answers are generated per unanswerable question (configurable in the code)
