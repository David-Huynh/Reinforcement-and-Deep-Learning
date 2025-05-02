import pickle

with open("./logs/models/best8/eval.pkl", "rb") as f:
    eval_metrics = pickle.load(f)

print(eval_metrics)