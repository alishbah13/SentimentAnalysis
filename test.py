import json
from model_train import predict_review

f = open("prob.txt", "r")
probabilities = json.load(f)
f.close()
f2 = open("range.txt" , "r")
range = json.load(f2)
f2.close()

User_input = str(input("Enter your review : " ))
p = predict_review(User_input , probabilities['log_prior'] , probabilities['log_likelihood'])

rating = (5-1) * ((p - (range["min"] / 5) ) / (range["max"]/5 - range["min"]/5 )) + 1

print("Prediction of review : " , p)
print("Rating of review   :  ",format(rating, '0.4f'))
