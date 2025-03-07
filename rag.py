# Import the necessary libraries.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import contextlib
import gc
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import streamlit as st
import datetime


# Create handlers for logging.
class PrintHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        print(log_entry)
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        st.write(log_entry)

# Configure logging including these custom handlers.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs_{timestamp}.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode='w'),  # log to file with timestamp
                        PrintHandler(),
                        # StreamlitHandler(),
                        ])


# Load the rules from a text file into a string.
def load_rules_from_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            rules_text = file.read()
        return rules_text
    except Exception as e:
        logging.error(f"Error loading rules from file: {e}")
        return ""

# Get a list of non-blank lines from the rules.
def preprocess_document(document):
    chunks = document.split('\n')  # split the document into lines
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # remove empty lines
    return chunks


# Run a retriever model from a provided question and full context.
def run_retriever(question, chunks, model_name, use_gpu_if_available=True, top_n=5):

    # If the question is None, return None.
    if question is None:
        return None
    
    # Determine the device to use for running the retriever.
    device = torch.device("cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu")
    logging.info(f"For retrieval we are using the device: {device}.")

    # Initialize the model.
    model = SentenceTransformer(model_name, device=device)

    # Encode the question into an embedding for the model.
    query_embedding = model.encode(question, convert_to_tensor=True)

    # Encode the chunks into embeddings for the model.
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Calculate the cosine similarities between the question and the chunks.
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

    # Move the similarities to CPU and convert to NumPy array.
    similarities = similarities.cpu().numpy()

    # Get the indices of the top N most relevant chunks to the question.
    relevant_indices = np.argsort(similarities)[-top_n:][::-1]

    # Get standard indices for the chunks.
    index = {i: chunk for i, chunk in enumerate(chunks)}

    # Get the most relevant lines.
    context = [index[i] for i in relevant_indices]

    # Clear memory appropriately.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Return the most relevant lines as context for the prompt.
    return context


def assemble_prompt(context, question):

    if context is None:
        return question

    # Combine context and question into a single prompt.
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "

    return prompt


# Define a null context manager.
@contextlib.contextmanager
def null_context():
    yield


# Run a generator model from a provided prompt.
def run_generator(prompt, model_name, use_gpu_if_available=True, mixed_precision=False, load_in_4bit=False, max_new_tokens=10, num_return_sequences=1, temperature=0.3, top_k=5, do_sample=True):

    # Determine the device to use for running the generator.
    device = torch.device("cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu")
    logging.info(f"For generation we are using the device: {device}.")

    # Configure the compute precision.
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None

    # Load the pre-trained model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the input into numerical interget token IDs (input_ids attribute).
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Determine the context to use for mixed precision.
    context_to_use = torch.autocast("cuda") if mixed_precision and device.type == "cuda" else null_context()

    # Generate the answer with the desired precision.
    with context_to_use:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,  # limit the number of new tokens generated
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.ne(tokenizer.pad_token_id).long(),  # specify the attention mask
            temperature=temperature,  # lower temperature for more deterministic output
            top_k=top_k,  # limit the number of possible next tokens
            do_sample=do_sample,  # enable sampling
        )

    # Decode the model output from the output token IDs to text.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clear memory appropriately.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Return the response.
    return response


# Define the main function.
def main():

    # Parameters.
    rules_filename = "ultimate_frisbee_rules-manual_copy_from_website-edited.txt"
    # question = "Explain the timeout rules"
    question = "What is the stall count?"
    retriever_model_name = 'all-MiniLM-L6-v2'
    generator_model_name = 'gpt2'
    # generator_model_name = 'gpt2-xl'  # this is the largest GPT-2 model from OpenAI and is open source
    # generator_model_name = "EleutherAI/gpt-neo-2.7B"  # best GPT-related model for my laptop
    # generator_model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'  # works, best deepseek model I can get working
    top_k_for_retriever = 5
    use_gpu_if_available = True
    mixed_precision = False
    load_in_4bit = False  # a bit faster for gpt-neo but sometimes gives wrong answers, sometimes repeats the question, but works pretty reliably for the simple example
    max_new_tokens = 10
    num_return_sequences = 1
    temperature = 0.3
    top_k_for_generator = 5
    do_sample = True



    # Load the rules from a text file into a string.
    rules_text = load_rules_from_file_to_string(rules_filename)

    # Get a list of non-blank lines from the rules.
    full_context_chunks = preprocess_document(rules_text)

    # Run the retriever to obtain context for the prompt.
    context = run_retriever(question, full_context_chunks, retriever_model_name, use_gpu_if_available=use_gpu_if_available, top_n=top_k_for_retriever)

    # Assemble the prompt for the generator.
    prompt = assemble_prompt(context, question)
    # prompt = "Andrew's favorite color is violet. What is Andrew's favorite color?"

    # Run the generator with the provided prompt.
    response = run_generator(prompt, generator_model_name, use_gpu_if_available=use_gpu_if_available, mixed_precision=mixed_precision, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=temperature, top_k=top_k_for_generator, do_sample=do_sample)

    # Print the generator's response.
    logging.info(response)


# Execute the main function.
if __name__ == "__main__":
    main()
