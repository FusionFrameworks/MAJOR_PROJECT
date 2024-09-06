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




from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import os
from googletrans import Translator

app = Flask(__name__)
app.secret_key = '7888accc4a8508133c302792b0cd27f471bb262bc5666c'

# Sample training data with diseases and their symptoms in English, Kannada, and Hindi
training_data = [
    ("Fever", "high temperature, chills, sweating, headache, muscle pain"),
    ("Flu", "fever, cough, sore throat, runny or stuffy nose, muscle aches"),
    ("Cold", "sore throat, runny nose, coughing, sneezing, mild fever"),
    ("Headache", "dull pain, pressure in the head, nausea, sensitivity to light"),
    ("Stomachache", "abdominal pain, bloating, nausea, vomiting, diarrhea"),
    ("Allergy", "sneezing, itchy eyes, runny nose, rash, swelling"),

    ("ಜ್ವರ", "ಹೆಚ್ಚಿನ ತಾಪಮಾನ, ಚಿಲ್ಲುಗಳು, ಸ್ವೇತ, ತಲೆನೋವು, ಸ್ನಾಯು ನೋವು"),
    ("ಫ್ಲು", "ಜ್ವರ, ಕಫ, ತುಕ್ಕು, ಓಡೋಣ ಅಥವಾ ನಾಕೆ, ಸ್ನಾಯು ನೋವು"),
    ("ಚಳಿ", "ತಲೆನೋವು, ಓಡೋಣ, ಕೆಮ್ಮು, ಚಿರತೆ, ಕಡಿಮೆ ಜ್ವರ"),
    ("ತಲೆನೋವು", "ಮಂದ ಪೆಡ, ತಲೆಯೊಳಗಿನ ಒತ್ತಣೆ, ವಾಂತಿ, ಬೆಳಕುಗೆ ಅಸಹನಶೀಲತೆ"),
    ("ಉಸಿರಾಟದ ನೋವು", "ಪೆಟ್ನ ನೋವು, ಬ್ಲೋಟಿಂಗ್, ವಾಂತಿ, ಉಸಿರಾಟದ ನೋವು, ಜೀರ್ಣಶಕ್ತಿ"),
    ("ಆಲರ್ಜೀ", "ಚಿರತೆ, ಕಣ್ಣುಗಳು ಸೀಡು, ಓಡೋಣ, ಉಬ್ಬು, ಪುಟ್ಟ"),

    ("बुखार", "उच्च तापमान, ठंड लगना, पसीना, सिरदर्द, मांसपेशियों में दर्द"),
    ("फ्लू", "ज्वर, खांसी, गले में खराश, बहती या बंद नाक, मांसपेशियों में दर्द"),
    ("सर्दी", "गले में खराश, बहती नाक, खांसी, छींके, हल्का ज्वर"),
    ("सिरदर्द", "सुस्त दर्द, सिर में दबाव, मतली, प्रकाश के प्रति संवेदनशीलता"),
    ("पेट दर्द", "पेट में दर्द, सूजन, मतली, उल्टी, दस्त"),
    ("एलर्जी", "छींके, खुजली वाली आँखें, बहती नाक, चकत्ते, सूजन")
]

# Create a dictionary for symptoms to diseases mapping in English, Kannada, and Hindi
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
    "swelling": "Allergy",

    "ಹೆಚ್ಚಿನ ತಾಪಮಾನ": "ಜ್ವರ",
    "ಚಿಲ್ಲುಗಳು": "ಜ್ವರ",
    "ಸ್ವೇತ": "ಜ್ವರ",
    "ತಲೆನೋವು": "ಜ್ವರ",
    "ಸ್ನಾಯು ನೋವು": "ಜ್ವರ",
    "ಕೆಮ್ಮು": "ಫ್ಲು",
    "ತುಕ್ಕು": "ಫ್ಲು",
    "ಓಡೋಣ ಅಥವಾ ನಾಕೆ": "ಫ್ಲು",
    "ಸ್ನಾಯು ನೋವು": "ಫ್ಲು",
    "ತಲೆನೋವು": "ಚಳಿ",
    "ಓಡೋಣ": "ಚಳಿ",
    "ಕೆಮ್ಮು": "ಚಳಿ",
    "ಚಿರತೆ": "ಚಳಿ",
    "ಕಡಿಮೆ ಜ್ವರ": "ಚಳಿ",
    "ಮಂದ ಪೆಡ": "ತಲೆನೋವು",
    "ತಲೆಯೊಳಗಿನ ಒತ್ತಣೆ": "ತಲೆನೋವು",
    "ವಾಂತಿ": "ತಲೆನೋವು",
    "ಬಳಕುಗಾಗಿ ಅಸಹನಶೀಲತೆ": "ತಲೆನೋವು",
    "ಪೆಟ್ನ ನೋವು": "ಉಸಿರಾಟದ ನೋವು",
    "ಬ್ಲೋಟಿಂಗ್": "ಉಸಿರಾಟದ ನೋವು",
    "ವಾಂತಿ": "ಉಸಿರಾಟದ ನೋವು",
    "ಉಸಿರಾಟದ ನೋವು": "ಉಸಿರಾಟದ ನೋವು",
    "ಜೀರ್ಣಶಕ್ತಿ": "ಉಸಿರಾಟದ ನೋವು",
    "ಚಿರತೆ": "ಆಲರ್ಜೀ",
    "ಕಣ್ಣುಗಳು ಸೀಡು": "ಆಲರ್ಜೀ",
    "ಓಡೋಣ": "ಆಲರ್ಜೀ",
    "ಉಬ್ಬು": "ಆಲರ್ಜೀ",
    "ಪುಟ್ಟ": "ಆಲರ್ಜೀ",

    "उच्च तापमान": "बुखार",
    "ठंड लगना": "बुखार",
    "पसीना": "बुखार",
    "सिरदर्द": "बुखार",
    "मांसपेशियों में दर्द": "बुखार",
    "खांसी": "फ्लू",
    "गले में खराश": "फ्लू",
    "बहती या बंद नाक": "फ्लू",
    "मांसपेशियों में दर्द": "फ्लू",
    "गले में खराश": "सर्दी",
    "बहती नाक": "सर्दी",
    "खांसी": "सर्दी",
    "छींके": "सर्दी",
    "हल्का ज्वर": "सर्दी",
    "सुस्त दर्द": "सिरदर्द",
    "सिर में दबाव": "सिरदर्द",
    "मतली": "सिरदर्द",
    "प्रकाश के प्रति संवेदनशीलता": "सिरदर्द",
    "पेट में दर्द": "पेट दर्द",
    "सूजन": "पेट दर्द",
    "मतली": "पेट दर्द",
    "उल्टी": "पेट दर्द",
    "दस्त": "पेट दर्द",
    "छींके": "एलर्जी",
    "खुजली वाली आँखें": "एलर्जी",
    "बहती नाक": "एलर्जी",
    "चकत्ते": "एलर्जी",
    "सूजन": "एलर्जी"
}

# Create a dictionary for disease names in Kannada and Hindi
disease_names = {
    "Fever": {"kn": "ಜ್ವರ", "hi": "बुखार"},
    "Flu": {"kn": "ಫ್ಲು", "hi": "फ्लू"},
    "Cold": {"kn": "ಚಳಿ", "hi": "सर्दी"},
    "Headache": {"kn": "ತಲೆನೋವು", "hi": "सिरदर्द"},
    "Stomachache": {"kn": "ಉಸಿರಾಟದ ನೋವು", "hi": "पेट दर्द"},
    "Allergy": {"kn": "ಆಲರ್ಜೀ", "hi": "एलर्जी"}
}

translator = Translator()

# Function to train and save the model
def train_and_save_model():
    # Split the training data into inputs and responses
    X_train, y_train = zip(*training_data[:6])  # Use only the English data for training

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
    user_language = request.json.get('language', 'en')

    if user_language == 'kn':
        user_message = translator.translate(user_message, src='kn', dest='en').text
    elif user_language == 'hi':
        user_message = translator.translate(user_message, src='hi', dest='en').text

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
        response = f"I think you might have: {', '.join(diseases)}. We are connecting you to a suitable doctor please wait."

    if user_language == 'kn':
        response = translator.translate(response, src='en', dest='kn').text
    elif user_language == 'hi':
        response = translator.translate(response, src='en', dest='hi').text

    session['conversation'].append(('bot', response))
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)






