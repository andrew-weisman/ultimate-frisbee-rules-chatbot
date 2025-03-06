import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib

# Define a null context manager
@contextlib.contextmanager
def null_context():
    yield

def ask_question(model_name, question, context, use_gpu_if_available=True, mixed_precision=False):
    # Load pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Combine context and question into a single prompt
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    context_to_use = torch.autocast("cuda") if mixed_precision and device.type == "cuda" else null_context()

    # Generate the answer with mixed precision
    with context_to_use:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,  # Limit the number of new tokens generated
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
            temperature=0.3,  # Lower temperature for more deterministic output
            top_k=5,  # Limit the number of possible next tokens
            do_sample=True  # Enable sampling
        )

    # Decode the generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer part from the generated text
    answer = answer.split("Answer:")[1].strip().split('.')[0]  # Extract the first sentence
    return answer

# Example usage
model_name = "EleutherAI/gpt-neo-2.7B"  # Change this to try different models
use_gpu_if_available = False
mixed_precision = False
context = "Andrew's favorite color is violet."
question = "What is Andrew's favorite color?"
answer = ask_question(model_name, question, context, use_gpu_if_available, mixed_precision)
print(answer)  # Should print "Violet."