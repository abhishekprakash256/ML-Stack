from transformers import pipeline

# Create a text generation pipeline
text_gen_pipeline = pipeline("text-generation", model="distilgpt2")

# Example prompt
prompt = "Once upon a time, in a land far, far away,"

# Generate text
result = text_gen_pipeline(prompt, max_length=100)
print(result)
