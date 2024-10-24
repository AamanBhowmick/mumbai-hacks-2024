from flask import Flask, render_template, redirect, request, jsonify, session
from firebase import firebase
import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
load_dotenv()
import os

cred = credentials.Certificate('./credential.json')  # Replace with your actual path
firebase_admin.initialize_app(cred)

db = firestore.client()
users_ref = db.collection('Users')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') 

@app.route("/")
def home():
    user = session.get('user')
    return render_template("index.html", user = user)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = users_ref.where('email', '==', email).stream()
        
        user_found = False
        for user in users:
            user_data = user.to_dict()
            
            # Use check_password_hash to verify hashed password
            if check_password_hash(user_data['password'], password):
                user_found = True
                session['user'] = user_data['email']
                return redirect("/")
            
        if not user_found:
            return redirect("/login")
    return render_template("login.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    add_users = users_ref.document()
    if request.method == 'POST':
        
        add_users.set({
            "email" : request.form.get('email'),
            "password" : generate_password_hash(request.form.get('password'))
        })
        return redirect("/login") 
    return render_template("register.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('user', None)  
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)