import json
from model_train import predict_review


f = open("prob.txt", "r")
probabilities = json.load(f)

print(format( predict_review("very nice and fantastic lovely blouse" , probabilities['log_prior'] , probabilities['log_likelihood'] ), '0.4f'))