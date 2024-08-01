# from flask import Flask, request, jsonify, render_template
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# import pickle
# import os

# app = Flask(__name__)

# # Sample training data with diseases and their symptoms
# training_data = [
#     ("Fever", "high temperature, chills, sweating, headache, muscle pain"),
#     ("Flu", "fever, cough, sore throat, runny or stuffy nose, muscle aches"),
#     ("Cold", "sore throat, runny nose, coughing, sneezing, mild fever"),
#     ("Headache", "dull pain, pressure in the head, nausea, sensitivity to light"),
#     ("Stomachache", "abdominal pain, bloating, nausea, vomiting, diarrhea"),
#     ("Allergy", "sneezing, itchy eyes, runny nose, rash, swelling")
# ]

# # Create a dictionary for symptoms to diseases mapping
# symptoms_to_disease = {
#     "high temperature": "Fever",
#     "chills": "Fever",
#     "sweating": "Fever",
#     "headache": "Fever",
#     "muscle pain": "Fever",
#     "cough": "Flu",
#     "sore throat": "Flu",
#     "runny or stuffy nose": "Flu",
#     "muscle aches": "Flu",
#     "sore throat": "Cold",
#     "runny nose": "Cold",
#     "coughing": "Cold",
#     "sneezing": "Cold",
#     "mild fever": "Cold",
#     "dull pain": "Headache",
#     "pressure in the head": "Headache",
#     "nausea": "Headache",
#     "sensitivity to light": "Headache",
#     "abdominal pain": "Stomachache",
#     "bloating": "Stomachache",
#     "nausea": "Stomachache",
#     "vomiting": "Stomachache",
#     "diarrhea": "Stomachache",
#     "sneezing": "Allergy",
#     "itchy eyes": "Allergy",
#     "runny nose": "Allergy",
#     "rash": "Allergy",
#     "swelling": "Allergy"
# }

# # Function to train and save the model
# def train_and_save_model():
#     # Split the training data into inputs and responses
#     X_train, y_train = zip(*training_data)

#     # Create the TF-IDF vectorizer
#     vectorizer = TfidfVectorizer()

#     # Transform the training data into feature vectors
#     X_train_vectors = vectorizer.fit_transform(X_train)

#     # Create and train the LinearSVC model
#     model = LinearSVC()
#     model.fit(X_train_vectors, y_train)

#     # Save the vectorizer and the model to disk
#     with open("chatbot_vectorizer.pkl", "wb") as f:
#         pickle.dump(vectorizer, f)

#     with open("chatbot_model.pkl", "wb") as f:
#         pickle.dump(model, f)

# # Train and save the model if not already saved
# if not os.path.exists("chatbot_vectorizer.pkl") or not os.path.exists("chatbot_model.pkl"):
#     train_and_save_model()

# # Load the vectorizer and model from disk
# with open("chatbot_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open("chatbot_model.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get("message")
#     if not user_input:
#         return jsonify({"error": "No input provided"}), 400

#     # Check if user input contains symptoms
#     symptoms = user_input.split(',')  # Assuming symptoms are separated by commas
#     symptoms = [symptom.strip() for symptom in symptoms]  # Clean up spaces

#     # Map symptoms to disease
#     predicted_disease = "Unknown"
#     for symptom in symptoms:
#         if symptom in symptoms_to_disease:
#             predicted_disease = symptoms_to_disease[symptom]
#             break

#     if predicted_disease != "Unknown":
#         response = f"Based on the symptoms provided, the predicted disease is: {predicted_disease}"
#     else:
#         # Transform the user input using the vectorizer
#         user_input_vector = vectorizer.transform([user_input])
#         # Predict the response using the model
#         response = model.predict(user_input_vector)[0]

#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template, session
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# import pickle
# import os

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Change this to a secret key for session management

# # Sample training data with diseases and their symptoms
# training_data = [
#     ("Fever", "high temperature, chills, sweating, headache, muscle pain"),
#     ("Flu", "fever, cough, sore throat, runny or stuffy nose, muscle aches"),
#     ("Cold", "sore throat, runny nose, coughing, sneezing, mild fever"),
#     ("Headache", "dull pain, pressure in the head, nausea, sensitivity to light"),
#     ("Stomachache", "abdominal pain, bloating, nausea, vomiting, diarrhea"),
#     ("Allergy", "sneezing, itchy eyes, runny nose, rash, swelling")
# ]

# # Create a dictionary for symptoms to diseases mapping
# symptoms_to_disease = {
#     "high temperature": "Fever",
#     "chills": "Fever",
#     "sweating": "Fever",
#     "headache": "Fever",
#     "muscle pain": "Fever",
#     "cough": "Flu",
#     "sore throat": "Flu",
#     "runny or stuffy nose": "Flu",
#     "muscle aches": "Flu",
#     "sore throat": "Cold",
#     "runny nose": "Cold",
#     "coughing": "Cold",
#     "sneezing": "Cold",
#     "mild fever": "Cold",
#     "dull pain": "Headache",
#     "pressure in the head": "Headache",
#     "nausea": "Headache",
#     "sensitivity to light": "Headache",
#     "abdominal pain": "Stomachache",
#     "bloating": "Stomachache",
#     "nausea": "Stomachache",
#     "vomiting": "Stomachache",
#     "diarrhea": "Stomachache",
#     "sneezing": "Allergy",
#     "itchy eyes": "Allergy",
#     "runny nose": "Allergy",
#     "rash": "Allergy",
#     "swelling": "Allergy"
# }

# # Function to train and save the model
# def train_and_save_model():
#     # Split the training data into inputs and responses
#     X_train, y_train = zip(*training_data)

#     # Create the TF-IDF vectorizer
#     vectorizer = TfidfVectorizer()

#     # Transform the training data into feature vectors
#     X_train_vectors = vectorizer.fit_transform(X_train)

#     # Create and train the LinearSVC model
#     model = LinearSVC()
#     model.fit(X_train_vectors, y_train)

#     # Save the vectorizer and the model to disk
#     with open("chatbot_vectorizer.pkl", "wb") as f:
#         pickle.dump(vectorizer, f)

#     with open("chatbot_model.pkl", "wb") as f:
#         pickle.dump(model, f)

# # Train and save the model if not already saved
# if not os.path.exists("chatbot_vectorizer.pkl") or not os.path.exists("chatbot_model.pkl"):
#     train_and_save_model()

# # Load the vectorizer and model from disk
# with open("chatbot_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open("chatbot_model.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get("message")
#     if not user_input:
#         return jsonify({"error": "No input provided"}), 400

#     # Initialize session if not already initialized
#     if 'asking_for_symptoms' not in session:
#         session['asking_for_symptoms'] = False

#     if session['asking_for_symptoms']:
#         # Process the symptoms
#         symptoms = user_input.split(',')  # Assuming symptoms are separated by commas
#         symptoms = [symptom.strip() for symptom in symptoms]  # Clean up spaces

#         # Map symptoms to disease
#         predicted_disease = "Unknown"
#         for symptom in symptoms:
#             if symptom in symptoms_to_disease:
#                 predicted_disease = symptoms_to_disease[symptom]
#                 break

#         if predicted_disease != "Unknown":
#             response = f"Based on the symptoms provided, the predicted disease is: {predicted_disease}"
#             # End the symptom collection phase
#             session['asking_for_symptoms'] = False
#         else:
#             response = "I couldn't determine the disease. Can you provide more symptoms?"

#     else:
#         # Regular chatbot response
#         if user_input.lower() in ["symptoms", "i have symptoms", "tell me symptoms"]:
#             response = "Please provide your symptoms separated by commas, e.g., 'fever, cough'."
#             session['asking_for_symptoms'] = True
#         else:
#             # Transform the user input using the vectorizer
#             user_input_vector = vectorizer.transform([user_input])
#             # Predict the response using the model
#             response = model.predict(user_input_vector)[0]

#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)




# from flask import Flask, request, jsonify, render_template, session
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# import pickle
# import os

# app = Flask(__name__)
# app.secret_key = '7888accc4a8508133c302792b0cd27f471bb262bc5666c'  # Change this to a secret key for session management

# # Sample training data with diseases and their symptoms
# training_data = [
#     ("Fever", "high temperature, chills, sweating, headache, muscle pain"),
#     ("Flu", "fever, cough, sore throat, runny or stuffy nose, muscle aches"),
#     ("Cold", "sore throat, runny nose, coughing, sneezing, mild fever"),
#     ("Headache", "dull pain, pressure in the head, nausea, sensitivity to light"),
#     ("Stomachache", "abdominal pain, bloating, nausea, vomiting, diarrhea"),
#     ("Allergy", "sneezing, itchy eyes, runny nose, rash, swelling")
# ]

# # Create a dictionary for symptoms to diseases mapping
# symptoms_to_disease = {
#     "high temperature": "Fever",
#     "chills": "Fever",
#     "sweating": "Fever",
#     "headache": "Fever",
#     "muscle pain": "Fever",
#     "cough": "Flu",
#     "sore throat": "Flu",
#     "runny or stuffy nose": "Flu",
#     "muscle aches": "Flu",
#     "sore throat": "Cold",
#     "runny nose": "Cold",
#     "coughing": "Cold",
#     "sneezing": "Cold",
#     "mild fever": "Cold",
#     "dull pain": "Headache",
#     "pressure in the head": "Headache",
#     "nausea": "Headache",
#     "sensitivity to light": "Headache",
#     "abdominal pain": "Stomachache",
#     "bloating": "Stomachache",
#     "nausea": "Stomachache",
#     "vomiting": "Stomachache",
#     "diarrhea": "Stomachache",
#     "sneezing": "Allergy",
#     "itchy eyes": "Allergy",
#     "runny nose": "Allergy",
#     "rash": "Allergy",
#     "swelling": "Allergy"
# }

# # Function to train and save the model
# def train_and_save_model():
#     # Split the training data into inputs and responses
#     X_train, y_train = zip(*training_data)

#     # Create the TF-IDF vectorizer
#     vectorizer = TfidfVectorizer()

#     # Transform the training data into feature vectors
#     X_train_vectors = vectorizer.fit_transform(X_train)

#     # Create and train the LinearSVC model
#     model = LinearSVC()
#     model.fit(X_train_vectors, y_train)

#     # Save the vectorizer and the model to disk
#     with open("chatbot_vectorizer.pkl", "wb") as f:
#         pickle.dump(vectorizer, f)

#     with open("chatbot_model.pkl", "wb") as f:
#         pickle.dump(model, f)

# # Check if the model files exist, if not, train and save the model
# if not os.path.exists("chatbot_vectorizer.pkl") or not os.path.exists("chatbot_model.pkl"):
#     train_and_save_model()

# # Load the vectorizer and the model from disk
# with open("chatbot_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open("chatbot_model.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json['message']
#     session.setdefault('conversation', []).append(('user', user_message))

#     # Transform the user input into a feature vector
#     user_message_vector = vectorizer.transform([user_message])

#     # Make a prediction using the trained model
#     response = model.predict(user_message_vector)[0]

#     # Predict disease based on symptoms
#     symptoms = user_message.lower().split(", ")
#     diseases = set()
#     for symptom in symptoms:
#         if symptom in symptoms_to_disease:
#             diseases.add(symptoms_to_disease[symptom])
#     if diseases:
#         response = f"I think you might have: {', '.join(diseases)}. Please provide the phone number for further assistance."

#     session['conversation'].append(('bot', response))
#     return jsonify({'response': response})

# @app.route('/call', methods=['POST'])
# def call():
#     phone_number = request.json['phone_number']
#     # Here you would integrate with an API to make a phone call
#     # For demonstration, we'll just return a success message
#     response = f"Initiating call to {phone_number}. Our specialist will assist you shortly."
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


# FINALCODE


from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import os

app = Flask(__name__)
app.secret_key = '7888accc4a8508133c302792b0cd27f471bb262bc5666c'  # Change this to a secret key for session management

# Sample training data with diseases and their symptoms
training_data = [
    ("Fever", "high temperature, chills, sweating, headache, muscle pain"),
    ("Flu", "fever, cough, sore throat, runny or stuffy nose, muscle aches"),
    ("Cold", "sore throat, runny nose, coughing, sneezing, mild fever"),
    ("Headache", "dull pain, pressure in the head, nausea, sensitivity to light"),
    ("Stomachache", "abdominal pain, bloating, nausea, vomiting, diarrhea"),
    ("Allergy", "sneezing, itchy eyes, runny nose, rash, swelling")
]

# Create a dictionary for symptoms to diseases mapping
symptoms_to_disease = {
    "high temperature": "Fever",
    "chills": "Fever",
    "sweating": "Fever",
    "headache": "Fever",
    "muscle pain": "Fever",
    "cough": "Flu",
    "sore throat": "Flu",
    "runny or stuffy nose": "Flu",
    "muscle aches": "Flu",
    "sore throat": "Cold",
    "runny nose": "Cold",
    "coughing": "Cold",
    "sneezing": "Cold",
    "mild fever": "Cold",
    "dull pain": "Headache",
    "pressure in the head": "Headache",
    "nausea": "Headache",
    "sensitivity to light": "Headache",
    "abdominal pain": "Stomachache",
    "bloating": "Stomachache",
    "nausea": "Stomachache",
    "vomiting": "Stomachache",
    "diarrhea": "Stomachache",
    "sneezing": "Allergy",
    "itchy eyes": "Allergy",
    "runny nose": "Allergy",
    "rash": "Allergy",
    "swelling": "Allergy"
}

# Function to train and save the model
def train_and_save_model():
    # Split the training data into inputs and responses
    X_train, y_train = zip(*training_data)

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the training data into feature vectors
    X_train_vectors = vectorizer.fit_transform(X_train)

    # Create and train the LinearSVC model
    model = LinearSVC()
    model.fit(X_train_vectors, y_train)

    # Save the vectorizer and the model to disk
    with open("chatbot_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("chatbot_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Check if the model files exist, if not, train and save the model
if not os.path.exists("chatbot_vectorizer.pkl") or not os.path.exists("chatbot_model.pkl"):
    train_and_save_model()

# Load the vectorizer and the model from disk
with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    session.setdefault('conversation', []).append(('user', user_message))

    # Transform the user input into a feature vector
    user_message_vector = vectorizer.transform([user_message])

    # Make a prediction using the trained model
    response = model.predict(user_message_vector)[0]

    # Predict disease based on symptoms
    symptoms = user_message.lower().split(", ")
    diseases = set()
    for symptom in symptoms:
        if symptom in symptoms_to_disease:
            diseases.add(symptoms_to_disease[symptom])
    if diseases:
        response = f"I think you might have: {', '.join(diseases)}. Please provide the phone number for further assistance."

    session['conversation'].append(('bot', response))
    return jsonify({'response': response})

@app.route('/call', methods=['POST'])
def call():
    phone_number = request.json['phone_number']
    # Here you would integrate with an API to make a phone call
    # For demonstration, we'll just return a success message
    response = f"Initiating call to {phone_number}. Our specialist will assist you shortly."
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)



