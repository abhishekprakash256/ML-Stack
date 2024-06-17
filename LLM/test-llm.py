"""
make the testin of the LLM model 
"""


from transformers import pipeline


sentiment_pipeline = pipeline("sentiment-analysis")
#data = ["I love you", "I hate you"]


data = "there is a deer that lives in the far mountains, I don't like the mountain but the deer is good to eat, I tried to kill the dear but it attacked me and I got injured but I am not injured much as I defended myself well"


print(sentiment_pipeline(data))

