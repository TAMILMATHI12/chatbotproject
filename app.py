from flask import Flask,render_template,request,redirect,url_for
import tflearn
import numpy as np
import pickle,random
import json
import sqlite3
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import ms as ms

app = Flask(__name__)

with open("assets/input_data.pickle", "rb") as f:
	words, labels, training, output = pickle.load(f)

with open("assets/intents.json") as myfile:
	data = json.load(myfile)

#tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])

network = tflearn.fully_connected(network,8)
network = tflearn.fully_connected(network,8)

network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.load("assets/model.chatbot.tflearn")

chats=[]
@app.route("/") #home
def hello():
	return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("form.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("form.html")
    else:
        return render_template("signup.html")

@app.route('/form')
def form():
    return render_template('form.html')

@app.route("/start",methods=['POST','GET'])
def start():
	inp = [str(x) for x in request.form.values()]
	print(inp[0])
	#return render_template('chat_bot.html',result=inp[0])
	results = model.predict([bag_of_words(inp[0],words)])[0] 
	print(results)
	results_index = np.argmax(results)
	tag = labels[results_index]
	print(tag)  

	if results[results_index] < 0.8 or len(inp[0])<2:
		email="sai2572000@gmail.com"
		Msg="Chat bot not able to answer for this question"+inp
		ms.process(email,Msg)
		result ="Sorry, I didn't get you. Please try again. Kindly make live chat to 6382621629"

				
	else:
		for tg in data['intents']:
			if tg['tag'] == tag:
				responses = tg['responses']

		result=""+random.choice(responses)
		es= ["Sad to see you go :(", "Talk to you later", "Goodbye!", "See you later","bye-bye"]
		if result in es:
			email="sai2572000@gmail.com"
			Msg="Chat bot not able to answer for this question"+str(inp[0])
			ms.process(email,Msg)
			result+="Kindly make live chat to 6382621629"
			#result ="Sorry, I didn't get you. Please try again."

	chats.append("You: " + inp[0])
	chats.append(result)
	return render_template('form.html',chats=chats[::-1],type="")

	
def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

			
# start() 
if __name__=="__main__":
	app.run(debug=True)

