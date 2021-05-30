import json
from model_train import predict_review


f = open("prob.txt", "r")
probabilities = json.load(f)
f.close()

print(format( predict_review("not" , probabilities['log_prior'] , probabilities['log_likelihood'] ), '0.4f'))