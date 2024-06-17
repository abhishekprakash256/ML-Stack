"""
make the testin of the LLM model 
"""

from transformers import pipeline

# Create the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example data
data = "there is a deer that lives in the far mountains, I don't like the mountain but the deer is good to eat, I tried to kill the deer but it attacked me and I got injured but I am not injured much as I defended myself well"

# Run the sentiment analysis
result = sentiment_pipeline(data)

# Print the result
print(result)

# Print out the model information
print(sentiment_pipeline.model)
