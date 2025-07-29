from transformers import pipeline

pipe = pipeline("text-classification")
print(pipe("This restaurant is awesome"))
