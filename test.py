import json
from model_train import predict_review

f = open("prob.txt", "r")
probabilities = json.load(f)
f.close()
f2 = open("range.txt" , "r")
range = json.load(f2)
f2.close()

p = predict_review("good blouse will surely buy again " , probabilities['log_prior'] , probabilities['log_likelihood'])

rating = (5-0) * ((p - range["min"] ) / (range["max"] - range["min"] )) + 0

print("Prediction of review : " , p)
print("Rating of review   :  ",format(rating, '0.4f'))
