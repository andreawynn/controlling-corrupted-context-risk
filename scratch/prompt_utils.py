import random


def get_prompt_fn(dataset):
    if dataset == 'financial_phrasebank':
        return get_financial_phrasebank_prompt
    elif dataset == 'sst2':
        return get_sst2_prompt
    elif dataset == 'unnatural':
        return get_unnatural_prompt
    elif dataset == 'tweeteval_atheism':
        return get_tweeteval_atheism_prompt
    elif dataset == 'tweeteval_hate':
        return get_tweeteval_hate_prompt
    elif dataset == 'tweeteval_feminist':
        return get_tweeteval_feminist_prompt
    elif dataset == 'boolean':
        return get_boolean_prompt
    elif dataset == 'navigation':
        return get_navigation_prompt
    elif dataset == 'sports':
        return get_sports_prompt
    elif dataset == 'web_of_lies':
        return get_web_of_lies_prompt


# For calibration prompts, replace question with the empty input, e.g. "N/A", "[MASK]" etc
def get_all_prompts_single_question(n_examples_per_class, text_only, labels, question, bad_labels_map, dataset):
    # Choose an equal number of examples of each class for demos
    examples = {}
    for label in bad_labels_map:
        other_questions = [text_only[i] for i in range(len(text_only)) if str(labels[i]).lower() == str(label).lower() and text_only[i] != question]
        if n_examples_per_class < len(other_questions):
            examples[label] = random.sample(other_questions, n_examples_per_class)
        else:
            # Sample with replacement if there are not enough examples
            examples[label] = random.choices(other_questions, k=n_examples_per_class)

    # Construct a single list containing all examples and their labels
    example_text, example_labels, bad_labels = [], [], []
    for i in range(n_examples_per_class):
        for label in bad_labels_map:
            example_text.append(examples[label][i])
            example_labels.append(str(label).lower())
            bad_labels.append(bad_labels_map[str(label).lower()])

    # Randomize order of the text and labels together
    c = list(zip(example_text, example_labels, bad_labels))
    random.shuffle(c)
    example_text, example_labels, bad_labels = zip(*c)

    # Retrieve the prompt with the examples given in this randomized order
    prompt_fn = get_prompt_fn(dataset)
    c, i, z = prompt_fn(question, example_text, example_labels), prompt_fn(question, example_text, bad_labels), prompt_fn(question, [], [], zeroshot=True)
    return c, i, z 


def get_financial_phrasebank_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given snippet of text. The possible classes are: positive, negative, neutral. "
    prompt += "Output only the class of the text snippet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of text-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_boolean_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the truth value of a given boolean expression. The possible values are: true, false. "
    prompt += "Output only the class of the expression and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of boolean expressions and their associated truth values. \n"
        for i in range(len(example_texts)):
            prompt += " Expression: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Expression: " + str(question_text) + " \nAnswer: "
    return prompt


def get_navigation_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to gauge whether, when following a given set of navigation directions, you would end up back at your starting point. "
    prompt += "The possible answers are: yes, no. "
    prompt += "Output only the yes/no answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of navigation directions and whether they would bring you back to your starting point. \n"
        for i in range(len(example_texts)):
            prompt += " Directions: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Directions: " + str(question_text) + " \nAnswer: "
    return prompt


def get_sports_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to gauge whether a given sentence is a plausible statement. The possible answers are: yes, no. "
    prompt += "Output only the yes/no answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of statements and whether they are plausible. \n"
        for i in range(len(example_texts)):
            prompt += " Statement: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Statement: " + str(question_text) + " \nAnswer: "
    return prompt


def get_web_of_lies_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to identify whether a specific person tells the truth, given statements about who lies and who is honest. "
    prompt += "The possible answers are: yes, no. "
    prompt += "Output only the yes/no answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few example scenarios and classifications of a particular person's honesty. \n"
        for i in range(len(example_texts)):
            prompt += " Scenario: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Scenario: " + str(question_text) + " \nAnswer: "
    return prompt
    

def get_unnatural_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the type of a given object. The possible classes are: plant/vegetable, sport, animal. "
    prompt += "Output only the class of the text snippet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of object-class pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Object: " + str(example_texts[i]) + " \nClass: " + str(example_answers[i]) + "\n"
    prompt += " Object: " + str(question_text) + " \nClass: "
    return prompt


def get_sst2_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given snippet of text. The possible classes are positive or negative. "
    prompt += "Output only the class of the text snippet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of text-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_tweeteval_hate_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given tweet. The possible classes are favor or against. "
    prompt += "Output only the class of the tweet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of tweet-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Tweet: " + str(example_texts[i]) + " \nSentiment: " + str(example_answers[i]) + "\n"
    prompt += " Tweet: " + str(question_text) + " \nSentiment: "
    return prompt


def get_tweeteval_atheism_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Determine if the text supports atheism. The possible answers are yes, no, or neither. "
    prompt += "Output only the answer and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_tweeteval_feminist_prompt(question_text, example_texts, example_answers, zeroshot=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Determine if the text supports feminism. The possible answers are yes, no, or neither. "
    prompt += "Output only the answer and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt