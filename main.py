import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 1: Load the rules text from the provided file
def load_rules(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        rules_text = file.read()
    return rules_text

def fine_tune_model(model, tokenizer, rules_text):
    inputs = tokenizer(rules_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.step()

def retrieve_relevant_section(query, vectorizer, rules_vectors, rules_sections):
    query_vector = vectorizer.transform([query])
    similarities = np.dot(query_vector, rules_vectors.T).toarray()[0]
    most_similar_index = np.argmax(similarities)
    return rules_sections[most_similar_index]

def ask_question(question):
    relevant_section = retrieve_relevant_section(question, vectorizer, rules_vectors, rules_sections)
    input_text = f"Context: {relevant_section}\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

rules_text = load_rules("ultimate_frisbee_rules-manual_copy_from_website.txt")

# Step 2: Preprocess the Text (if needed)
# Here you can add any text preprocessing steps if required

# Step 3: Fine-Tune GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

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
