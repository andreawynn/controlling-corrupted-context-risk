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
    elif dataset == 'ag_news':
        return get_ag_news_prompt
    elif dataset == 'mqp':  
        return get_mqp_prompt
    elif dataset == 'mrp':
        return get_mrp_prompt
    elif dataset == 'wnli':
        return get_wnli_prompt
    elif dataset == 'trec':
        return get_trec_prompt
    elif dataset == 'spam':
        return get_spam_prompt


# For calibration prompts, replace question with the empty input, e.g. "N/A", "[MASK]" etc
def get_all_prompts_single_question(n_examples_per_class, text_only, labels, question, bad_labels_map, dataset, fake_labels):
    # Choose an equal number of examples of each class for demos
    examples = {}
    for label in bad_labels_map:
        other_qs = [text_only[i] for i in range(len(text_only)) if str(labels[i]).lower() == str(label).lower() and text_only[i] != question]
        n_examples_per_class = min(n_examples_per_class, len(other_qs))
        examples[label] = random.sample(other_qs, n_examples_per_class)

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
    c, i = prompt_fn(question, example_text, example_labels, False, fake_labels), prompt_fn(question, example_text, bad_labels, False, fake_labels)
    z = prompt_fn(question, [], [], zeroshot=True, fake_labels=fake_labels)
    return c, i, z 


def get_ag_news_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Your job is to classify the topic of a news article given a description of the article. "
    if fake_labels:
        prompt += "Output river if the topic is world, stone if the topic is sports, cloud if the topic is business, "
        prompt += "and chair if the topic is science/technology. "
    else:
        prompt += "The possible topics are: world, sports, business, science/technology. "
    prompt += "Output only the topic of the news article and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of description-topic pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Description: " + str(example_texts[i]) + " \nTopic: " + str(example_answers[i]) + "\n"
    prompt += " Description: " + str(question_text) + " \nTopic: "
    return prompt


def get_trec_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Your job is to classify the type of a snippet of text. "
    if fake_labels:
        prompt += "Output river if the type is abbreviation, stone if the type is entity, cloud if the type is description, "
        prompt += "chair if the type is human, table if the type is location, and grass if the type is numeric. "
    else:
        prompt += "The possible types are: abbreviation, entity, description, human, location, numeric. "
    prompt += "Output only the type of the text and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of text-type pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nType: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nType: "
    return prompt


def get_spam_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Your job is to classify whether a given email is spam. "
    if fake_labels:
        prompt += "Output river if the email is spam and stone if the email is not spam. "
    else:
        prompt += "Answer yes if the email is spam and no otherwise. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of emails and their spam classifications. \n"
        for i in range(len(example_texts)):
            prompt += " Email: " + str(example_texts[i]) + " \nSpam: " + str(example_answers[i]) + "\n"
    prompt += " Email: " + str(question_text) + " \nSpam: "
    return prompt


def get_mqp_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Given a pair containing one medical question and a paraphrased version of the question prepared by a medical professional, "
    prompt += "your job is to identify whether the paraphrased versions are similar (syntactically dissimilar but contextually similar) "
    prompt += "or dissimilar (syntactically may look similar but contextually dissimilar). "
    if fake_labels:
        prompt += "Output river if the paraphrased version is similar, and stone if it is dissimilar. "
    else:
        prompt += "The possible answers are: similar, dissimilar. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of question-paraphrase pairs and associated answers. \n"
        for i in range(len(example_texts)):
            prompt += " Pair: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Pair: " + str(question_text) + " \nAnswer: "
    return prompt


def get_mrp_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Given a pair of sentences extracted from news sources on the web, identify whether the pair captures a paraphrase or "
    prompt += "semantic equivalence. "
    if fake_labels:
        prompt += "Output river if the paraphrased version is equivalent, and stone if it is different. "
    else:
        prompt += "The possible answers are: equivalent, different. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of sentence pairs and the answer to the semantic equivalence question. \n"
        for i in range(len(example_texts)):
            prompt += " Sentence Pair: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Sentence Pair: " + str(question_text) + " \nAnswer: "
    return prompt


def get_wnli_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    prompt = "Given a pair of sentences, identify whether or not the second sentence is an entailment of the first. "
    if fake_labels:
        prompt += "Output river if it is an entailment, and stone if it is not. "
    else:
        prompt += "The possible answers are: is entailment, not entailment. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of sentence pairs and the answer to the entailment question. \n"
        for i in range(len(example_texts)):
            prompt += " Sentence Pair: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Sentence Pair: " + str(question_text) + " \nAnswer: "
    return prompt


def get_financial_phrasebank_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given snippet of text. "
    if fake_labels:
        prompt += "Output river if the sentiment is positive, stone if the sentiment is negative, and cloud if the sentiment is neutral. "
    else:
        prompt += "The possible answers are: positive, negative, neutral. "
    prompt += "Output only the class of the text snippet and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of text-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]) + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_boolean_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the truth value of a given boolean expression. "
    if fake_labels:
        prompt += "Output river if the expression is true, and stone if the expression is false. "
    else:
        prompt += "The possible values are: true, false. "
    prompt += "Output only the class of the expression and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of boolean expressions and their associated truth values. \n"
        for i in range(len(example_texts)):
            prompt += " Expression: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Expression: " + str(question_text) + " \nAnswer: "
    return prompt


def get_navigation_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to gauge whether, when following a given set of navigation directions, you would end up back at your starting point. "
    if fake_labels:
        prompt += "Output river if the answer is yes and stone if the answer is no. "
    else:
        prompt += "The possible answers are: yes, no. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of navigation directions and whether they would bring you back to your starting point. \n"
        for i in range(len(example_texts)):
            prompt += " Directions: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Directions: " + str(question_text) + " \nAnswer: "
    return prompt


def get_sports_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to gauge whether a given sentence is a plausible statement. "
    if fake_labels:
        prompt += "Output river if the answer is yes and stone if the answer is no. "
    else:
        prompt += "The possible answers are: yes, no. "
    prompt += "Output only the answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of statements and whether they are plausible. \n"
        for i in range(len(example_texts)):
            prompt += " Statement: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Statement: " + str(question_text) + " \nAnswer: "
    return prompt


def get_web_of_lies_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to identify whether a specific person tells the truth, given statements about who lies and who is honest. "
    if fake_labels:
        prompt += "Output river if the answer is yes and stone if the answer is no. "
    else:
        prompt += "The possible answers are: yes, no. "
    prompt += "Output only the yes/no answer and nothing else. Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few example scenarios and classifications of a particular person's honesty. \n"
        for i in range(len(example_texts)):
            prompt += " Scenario: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Scenario: " + str(question_text) + " \nAnswer: "
    return prompt
    

def get_unnatural_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the type of a given object. "
    if fake_labels:
        prompt += "Output river if the class is plant/vegetable, stone if the class is sport, and cloud if the class is animal. "
    else:
        prompt += "The possible classes are: plant/vegetable, sport, animal. "
    prompt += "Output only the class of the text snippet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of object-class pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Object: " + str(example_texts[i]) + " \nClass: " + str(example_answers[i]).lower() + "\n"
    prompt += " Object: " + str(question_text) + " \nClass: "
    return prompt


def get_sst2_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given snippet of text. "
    if fake_labels:
        prompt += "Output river if the answer is positive and stone if the answer is negative. "
    else:
        prompt += "The possible answers are: positive, negative. "
    prompt += "Output only the sentiment of the text snippet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of text-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_tweeteval_hate_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given tweet. "
    if fake_labels:
        prompt += "Output river if the answer is favor and stone if the answer is against. "
    else:
        prompt += "The possible answers are: favor, against. "
    prompt += "Output only the sentiment of the tweet and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples of tweet-sentiment pairs. \n"
        for i in range(len(example_texts)):
            prompt += " Tweet: " + str(example_texts[i]) + " \nSentiment: " + str(example_answers[i]).lower() + "\n"
    prompt += " Tweet: " + str(question_text) + " \nSentiment: "
    return prompt


def get_tweeteval_atheism_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Determine if the text supports atheism. "
    if fake_labels:
        prompt += "Output river if the answer is yes, stone if the answer is no, and cloud if the answer is neither. "
    else:
        prompt += "The possible answers are: yes, no, neither. "
    prompt += "Output only the answer and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt


def get_tweeteval_feminist_prompt(question_text, example_texts, example_answers, zeroshot=False, fake_labels=False):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Determine if the text supports feminism. "
    if fake_labels:
        prompt += "Output river if the answer is yes, stone if the answer is no, and cloud if the answer is neither. "
    else:
        prompt += "The possible answers are: yes, no, neither. "
    prompt += "Output only the answer and nothing else. "
    prompt += "Do not provide chain of thought reasoning before your answer. "
    if not zeroshot:
        prompt += "Below are a few examples. \n"
        for i in range(len(example_texts)):
            prompt += " Text: " + str(example_texts[i]) + " \nAnswer: " + str(example_answers[i]).lower() + "\n"
    prompt += " Text: " + str(question_text) + " \nAnswer: "
    return prompt