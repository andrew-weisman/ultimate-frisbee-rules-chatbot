# Import the necessary libraries.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import contextlib
import gc


# Define a null context manager.
@contextlib.contextmanager
def null_context():
    yield


# Run a generator model from a provided prompt.
def run_generator(model_name, question, context, use_gpu_if_available=True, mixed_precision=False, load_in_4bit=False, max_new_tokens=10, num_return_sequences=1, temperature=0.3, top_k=5, do_sample=True):

    # Combine context and question into a single prompt.
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Configure the compute precision.
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None

    # Load pre-trained model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the input into numerical interget token IDs (input_ids attribute).
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move the model and its inputs to the desired device.
    device = torch.device("cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs.to(device)

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
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clear memory appropriately.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Return the response.
    return answer


# Example usage
# model_name = "EleutherAI/gpt-neo-2.7B"  # best GPT-related model for my laptop
# model_name = "EleutherAI/gpt-j-6B"  # won't work with any options even memory mapping (removed from code above)
# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'  # works, best deepseek model I can get working
# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'  # kernel crash
# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'  # kernel crash
model_name = 'gpt2-xl'  # this is the largest GPT-2 model from OpenAI and is open source
use_gpu_if_available = True
mixed_precision = False
load_in_4bit = False  # a bit faster for gpt-neo but sometimes gives wrong answers, sometimes repeats the question, but works pretty reliably for the simple example
context = "Andrew's favorite color is violet."
question = "What is Andrew's favorite color?"
# context = "Andrew's favorite color is violet. Laura\'s favorite color is very different from Andrew\'s."
# question = "What is Laura's favorite color?"
answer = run_generator(model_name, question, context, use_gpu_if_available, mixed_precision, load_in_4bit)
print(answer)  # Should print "Violet."
