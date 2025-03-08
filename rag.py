# Import the necessary libraries.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import contextlib
import gc
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import streamlit as st
from datetime import datetime
import time


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
if not logging.getLogger().hasHandlers():  # Check if handlers are already configured to avoid adding multiple handlers.
    logging.basicConfig(level=logging.DEBUG,
                        format='LOGGER: %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='w'),  # log to file with timestamp
                            PrintHandler(),
                            # StreamlitHandler(),
                            ])


# Define a function to get the desired computation device.
def get_desired_computation_device(computation_name="computation", use_gpu_if_available=True):
    if use_gpu_if_available and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")
        device_name = torch.cuda.get_device_name(device_index)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    logging.info(f"For {computation_name} we are using the device: {device_name}.")
    return device


# Define a function to run, log, and benchmark an arbitrary function.
def run_function(function, args=(), kwargs={}):
    logging.info(f">>>> Running function: {function.__name__}...")
    start_time = time.time()
    return_value = function(*args, **kwargs)
    logging.info(f"<<<< Function {function.__name__} completed in {time.time() - start_time:.2f} seconds.")
    return return_value


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
    device = get_desired_computation_device(computation_name="retrieval", use_gpu_if_available=use_gpu_if_available)

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
    del model, query_embedding, chunk_embeddings, similarities  # delete objects to free memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Return the most relevant lines as context for the prompt.
    return context


# Assemble the prompt for the generator.
def assemble_prompt(context, question):

    # Combine context and question into a single prompt.
    prompt = f"Context: {' '.join(context)}\nQuestion: {question}\nAnswer: "

    # Return the assembled prompt.
    return prompt


# Define a null context manager.
@contextlib.contextmanager
def null_context():
    yield


# Run a generator model from a provided prompt.
def run_generator(prompt, model_name, use_gpu_if_available=True, mixed_precision=False, load_in_4bit=False, max_new_tokens=10, num_return_sequences=1, temperature=0.3, top_k=5, do_sample=True):

    # Determine the device to use for running the generator.
    device = get_desired_computation_device(computation_name="generation", use_gpu_if_available=use_gpu_if_available)

    # Configure the compute precision.
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None

    # Load the pre-trained model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure that the padding tokens are treated as end-of-sequence tokens, which can help in generating more coherent responses and managing attention masks correctly. If things are slow, try commenting out the next line.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode the input into numerical interget token IDs (input_ids attribute). Copilot thinks it better to use this __call__ method instead of the specific encode method.
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
            attention_mask=inputs["input_ids"].ne(tokenizer.pad_token_id).long(),  # specify the attention mask
            temperature=temperature,  # lower temperature for more deterministic output
            top_k=top_k,  # limit the number of possible next tokens
            do_sample=do_sample,  # enable sampling
        )

    # Decode the model output from the output token IDs to text.
    response = "---\n"
    for ireturn_sequence in range(num_return_sequences):
        response += tokenizer.decode(outputs[ireturn_sequence], skip_special_tokens=True)
        response += "---\n"

    # Clear memory appropriately.
    del model, tokenizer, inputs, outputs  # delete objects to free memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Return the response.
    return response


# Define the main function.
def main():

    # Parameters.
    do_augmentation = False
    rules_filename = "ultimate_frisbee_rules-manual_copy_from_website-edited.txt"
    question = "Andrew's favorite color is violet. What is Andrew's favorite color?"
    # question = "Explain the timeout rules"
    # question = "What is the stall count?"
    retriever_model_name = 'all-MiniLM-L6-v2'
    generator_model_name = 'gpt2'
    # generator_model_name = 'gpt2-xl'  # this is the largest GPT-2 model from OpenAI and is open source
    # generator_model_name = "EleutherAI/gpt-neo-2.7B"  # best GPT-related model for my laptop
    # generator_model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'  # works, best deepseek model I can get working
    top_k_for_retriever = 5
    use_gpu_if_available = True
    mixed_precision = False
    load_in_4bit = False
    # max_new_tokens = 10
    max_new_tokens = None
    num_return_sequences = 10
    temperature = 0.3
    top_k_for_generator = 5
    do_sample = True

    # If we want to create a RAG and therefore include relevant context in the prompt...
    if do_augmentation:
    
        # Load the rules from a text file into a string.
        rules_text = run_function(load_rules_from_file_to_string, args=(rules_filename,))

        # Get a list of non-blank lines from the rules.
        full_context_chunks = run_function(preprocess_document, args=(rules_text,))

        # Run the retriever to obtain context for the prompt.
        context = run_function(run_retriever, args=(question, full_context_chunks, retriever_model_name), kwargs=dict(use_gpu_if_available=use_gpu_if_available, top_n=top_k_for_retriever))

        # Assemble the prompt for the generator.
        prompt = run_function(assemble_prompt, args=(context, question))

    # Otherwise, just use the question as the prompt:
    else:
        prompt = question

    # Run the generator with the provided prompt.
    response = run_function(run_generator, args=(prompt, generator_model_name), kwargs=dict(use_gpu_if_available=use_gpu_if_available, mixed_precision=mixed_precision, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=temperature, top_k=top_k_for_generator, do_sample=do_sample))

    # Print the generator's response.
    logging.info(response)


# Execute the main function.
if __name__ == "__main__":
    main()
