# This script demonstrates how to implement a question-answering system using a combination of retrieval and generation techniques.
# The system extracts text from a PDF, fine-tunes a GPT-2 model on the text, implements a retrieval mechanism using TF-IDF, and integrates retrieval with generation for question answering.

# Import necessary libraries
import fitz  # PyMuPDF
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


# Fine-Tune GPT-2 Model
def fine_tune_model(model, tokenizer, rules_text):
    inputs = tokenizer(rules_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.step()


# Retrieve Relevant Section
def retrieve_relevant_section(query, vectorizer, rules_vectors, rules_sections):
    query_vector = vectorizer.transform([query])
    similarities = np.dot(query_vector, rules_vectors.T).toarray()[0]
    most_similar_index = np.argmax(similarities)
    return rules_sections[most_similar_index]


# Function to ask questions
def ask_question(question):
    relevant_section = retrieve_relevant_section(question, vectorizer, rules_vectors, rules_sections)
    input_text = f"Context: {relevant_section}\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']


# Step 1: Extract text from the PDF
pdf_path = "ultimate_frisbee_rules.pdf"
rules_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a file
with open("ultimate_frisbee_rules.txt", "w") as file:
    file.write(rules_text)

# Step 2: Preprocess the Text (if needed)
# Here you can add any text preprocessing steps if required

# Step 3: Fine-Tune GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Fine-tune the model on the rules text
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
