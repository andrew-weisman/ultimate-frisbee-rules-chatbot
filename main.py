# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Step 1: Load the rules text from the provided file
def load_rules(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        rules_text = file.read()
    return rules_text


# Fine-tune the model using each line of the rules.
def fine_tune_model(model, tokenizer, rules_text):

    # Ensure the model is in training mode
    model.train()

    # Tokenize each line in the rules.
    tokens_per_line = [tokenizer(line, return_tensors="pt") for line in rules_text.split('\n')]

    # Get the number of tokens for each line of the rules.
    num_tokens_per_line = [len(tokens['input_ids'][0]) for tokens in tokens_per_line]

    # Ensure the model can handle every line.
    model_max_length = tokenizer.model_max_length
    assert max(num_tokens_per_line) <= model_max_length, 'The maximum number of tokens in all lines is greater than the maximum number of tokens the model can handle.'

    # Get the total number of lines i.e. number of token sets.
    tot_num_tokens = len(num_tokens_per_line)

    # Initialize the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # For each line in the rules...
    for itokens, tokens in enumerate(tokens_per_line):
        print(f'On line {itokens + 1} of {tot_num_tokens}: number of tokens: {len(tokens["input_ids"][0])}')
        outputs = model(**tokens, labels=tokens["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def retrieve_relevant_section(query, vectorizer, rules_vectors, rules_sections):
    query_vector = vectorizer.transform([query])
    similarities = np.dot(query_vector, rules_vectors.T).toarray()[0]
    most_similar_index = np.argmax(similarities)
    return rules_sections[most_similar_index]


def ask_question(question, vectorizer, rules_vectors, rules_sections, qa_pipeline):
    relevant_section = retrieve_relevant_section(question, vectorizer, rules_vectors, rules_sections)
    input_text = f"Context: {relevant_section}\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']


def main():

    rules_text = load_rules("ultimate_frisbee_rules-manual_copy_from_website.txt")

    # Step 2: Preprocess the Text (if needed)
    # Here you can add any text preprocessing steps if required

    # Step 3: Fine-Tune GPT-2
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)  # this downloads and caches (across sessions) some files including the model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # this also downloads and caches things

    fine_tune_model(model, tokenizer, rules_text)

    # Step 4: Implement Retrieval Mechanism
    # Split the rules text into sections for retrieval
    rules_sections = rules_text.split('\n\n')

    # Create a TF-IDF vectorizer and fit it on the rules sections
    vectorizer = TfidfVectorizer().fit(rules_sections)
    rules_vectors = vectorizer.transform(rules_sections)

    # Step 5: Integrate Retrieval with Generation
    # Create a pipeline for question answering
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Example question
    question = "What is the stall count in ultimate frisbee?"
    answer = ask_question(question)
    print(answer)


if __name__ == "__main__":
    main()
