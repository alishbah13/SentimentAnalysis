from tkinter import *
import json
from model_train import predict_review, rating

class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Product Review')
        self.lbl3=Label(win, text='Rating')
        self.lbl2=Label(win, text='Sentiment')
        self.t1=Entry()
        self.t3=Entry()
        self.t2=Entry()
        self.btn1 = Button(win, text='Submit')
        self.lbl1.place(x=50, y=50)
        self.t1.place(x=150, y=50, width=400)
        self.b1=Button(win, text='Submit', command=self.get_result)
        self.b1.place(x=470, y=100)
        self.lbl3.place(x=50, y=150)
        self.t3.place(x=150, y=150, width=400)
        self.lbl2.place(x=50, y=200)
        self.t2.place(x=150, y=200, width=400)
    def get_result(self):
        self.t3.delete(0, 'end')
        user_inp =str(self.t1.get())
        p = predict_review(user_inp, probabilities['log_prior'] , probabilities['log_likelihood'])
        result = rating(p, range['min'], range['max'])
        if p > 0:
            self.t2.insert(END, 'Positive Review')
        else:
            self.t2.insert(END, 'Negative Review')
        self.t3.insert(END, str(result))

f = open("prob.txt", "r")
probabilities = json.load(f)
f.close()

f2 = open("range.txt" , "r")
range = json.load(f2)
f2.close()

window=Tk()
mywin=MyWindow(window)
window.title('Sentiment Analyser')
window.geometry("600x250+0+0")
window.eval('tk::PlaceWindow . center')
window.mainloop()