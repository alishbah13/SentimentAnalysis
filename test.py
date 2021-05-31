import json
from model_train import predict_review, rating

f = open("prob.txt", "r")
probabilities = json.load(f)
f.close()
f2 = open("range.txt" , "r")
range = json.load(f2)
f2.close()

User_input = str(input("Enter your review : " ))
p = predict_review(User_input , probabilities['log_prior'] , probabilities['log_likelihood'])

rate = rating(p, range['min'], range['max'])

print("Prediction of review : " , p)
print("Rating of review   :  ",format(rate, '0.4f'))
