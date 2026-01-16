"""
Generate answers for unanswerable questions in SQuAD-v2 using GPT-5.

This script:
1. Loads the SQuAD-v2 dataset from HuggingFace
2. Filters to only unanswerable questions
3. Uses GPT-5 to generate valid answers for each unanswerable question
4. Saves only the unanswerable questions (with generated answers) in HuggingFace-compatible format
"""

import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv


def is_unanswerable(example):
    """Check if a question is unanswerable (empty answer text and answer_start)."""
    answers = example["answers"]
    return len(answers["text"]) == 0 and len(answers["answer_start"]) == 0


def generate_answers_with_gpt5(client, question, max_answers=5):
    """
    Use GPT-5 to generate valid short answers for a question, ignoring misleading context. 
    
    Args:
        client: OpenAI client
        question: The question to answer
    
    Returns:
        List of short answer strings
    """
    prompt = f"""Given the following question, generate all possible distinct, valid short answers to the question (up to {max_answers} total answers). 
                Each answer should be a brief phrase or span (typically 1-5 words) that directly answers the question.

                Question: {question}

                Provide one answer per line. Only output the answers, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise, accurate answers to questions."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=200
        )
        
        # Parse the response - split by newlines and clean up
        answer_text = response.choices[0].message.content.strip()
        answers = [ans.strip() for ans in answer_text.split("\n") if ans.strip()]
        
        # Remove any numbering if present (e.g., "1. answer" -> "answer")
        cleaned_answers = []
        for ans in answers:
            # Remove common numbering patterns
            if ans and len(ans) > 2:
                if ans[0].isdigit() and ans[1] in ".):":
                    ans = ans[2:].strip()
                elif ans[0].isdigit() and len(ans) > 3 and ans[1].isdigit() and ans[2] in ".):":
                    ans = ans[3:].strip()
            if ans:
                cleaned_answers.append(ans)
        
        return cleaned_answers[:max_answers]
        
    except Exception as e:
        print(f"Error calling GPT-5 API: {e}")
        return []


def process_unanswerable_questions(dataset_split, client, unanswerable_limit=None):
    """
    Process all unanswerable questions in a dataset split and return only those with generated answers.
    
    Args:
        dataset_split: A HuggingFace dataset split
        client: OpenAI client
        unanswerable_limit: Max number of unanswerable questions to generate answers for (None = all)
    
    Returns:
        Tuple of (dataset_with_only_unanswerable_questions, num_unanswerable_processed)
    """
    updated_data = []
    unanswerable_count = 0
    
    for example in tqdm(dataset_split, desc="Processing questions"):
        new_example = dict(example)
        
        if is_unanswerable(example):
            # Check if we've hit the limit
            if unanswerable_limit is not None and unanswerable_count >= unanswerable_limit:
                # Skip this unanswerable question (don't include in output)
                continue
            
            unanswerable_count += 1
            
            # Generate answers using GPT-5
            generated = generate_answers_with_gpt5(
                client,
                example["question"]
            )
            
            # Ensure we have at least 1 answers - retry if needed
            while len(generated) < 1:
                print(f"Got no answers for question: {example['question'][:50]}... Retrying...")
                generated = generate_answers_with_gpt5(
                    client,
                    example["question"]
                )
            
            # Build the answers dict with the GPT-generated answers
            new_answers = {
                "text": generated,  
                "answer_start": [0] * len(generated),  # Placeholder positions
                "is_unanswerable": True
            }
            
            new_example["answers"] = new_answers
            updated_data.append(new_example)
        # Skip answerable questions - only save unanswerable ones with generated answers
    
    return Dataset.from_list(updated_data), unanswerable_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for unanswerable SQuAD-v2 questions using GPT-5"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of unanswerable questions to process with GPT-5 (for testing). If not set, processes all."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./squad_v2_unanswerable_with_generated_answers",
        help="Output path for the processed dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "both"],
        default="both",
        help="Which dataset split(s) to process: 'train', 'validation', or 'both' (default: both)"
    )
    args = parser.parse_args()
    
    # Check for OpenAI API key
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-api-key'"
        )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    print("Loading SQuAD-v2 dataset from HuggingFace...")
    squad_v2 = load_dataset("squad_v2")
    
    # Count unanswerable questions
    train_unanswerable = sum(1 for ex in squad_v2["train"] if is_unanswerable(ex))
    val_unanswerable = sum(1 for ex in squad_v2["validation"] if is_unanswerable(ex))
    
    print(f"Training set: {len(squad_v2['train'])} total, {train_unanswerable} unanswerable")
    print(f"Validation set: {len(squad_v2['validation'])} total, {val_unanswerable} unanswerable")
    
    if args.limit:
        print(f"\n*** TEST MODE: Limiting to {args.limit} unanswerable questions per split ***\n")
    
    print(f"Split(s) to process: {args.split}")
    
    updated_splits = {}
    
    # Process training set if requested
    if args.split in ["train", "both"]:
        print("\nProcessing training set...")
        updated_train, train_processed = process_unanswerable_questions(
            squad_v2["train"], client, unanswerable_limit=args.limit
        )
        print(f"  Processed {train_processed} unanswerable questions")
        updated_splits["train"] = updated_train
    
    # Process validation set if requested
    if args.split in ["validation", "both"]:
        print("\nProcessing validation set...")
        updated_validation, val_processed = process_unanswerable_questions(
            squad_v2["validation"], client, unanswerable_limit=args.limit
        )
        print(f"  Processed {val_processed} unanswerable questions")
        updated_splits["validation"] = updated_validation
    
    # Create updated dataset dict
    updated_dataset = DatasetDict(updated_splits)
    
    # Save the dataset locally
    output_path = args.output
    print(f"\nSaving updated dataset to {output_path}...")
    updated_dataset.save_to_disk(output_path)
    
    print("Done! Dataset saved successfully.")
    print(f"\nTo load this dataset later, use:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_path}')")


if __name__ == "__main__":
    main()
