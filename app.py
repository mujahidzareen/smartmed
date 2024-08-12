from flask import Flask, request, jsonify
import json
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from openai import OpenAI
import prompts
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

keyy = os.getenv("keyy")
client = OpenAI(api_key=keyy)

main_prompt = prompts.system_message
message_history = [{"role": "user", "content": main_prompt}]
complition_history = []
message_for_extraction = []

# Load data and models
with open("release_evidences.json", "r") as file:
    data = json.load(file)

merged_evidences = json.load(open('merged_evidences.json'))
merged_conditions = json.load(open('merged_conditions.json'))

class ExtendedLabelEncoder(LabelEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def transform(self, y):
        unseen_labels = set(y) - set(self.classes_)
        if unseen_labels:
            self.classes_ = np.append(self.classes_, list(unseen_labels))
        return super().transform(y)

def load_model_and_encoder():
    try:
        pipeline = joblib.load('pathology_model.pkl')
        encoder_sex = joblib.load('encoder_sex.pkl')
    except FileNotFoundError:
        print("Model or encoder file not found. Please ensure the files are saved correctly.")
        return None, None
    return pipeline, encoder_sex

def predict_pathology(age, sex, symptoms, initial_evidence):
    pipeline, encoder_sex = load_model_and_encoder()
    if not pipeline or not encoder_sex:
        return None
    try:
        sex_encoded = encoder_sex.transform([sex])[0]
    except ValueError as e:
        print(f"Error encoding 'SEX': {e}")
        sex_encoded = encoder_sex.transform(['unknown'])[0] 
    
    user_data = pd.DataFrame({
        'AGE': [age],
        'SEX': [sex_encoded],
        'DIFFERENTIAL_DIAGNOSIS': ['no_data'],
        'EVIDENCES': [' '.join(symptoms) if symptoms else 'no_data'],
        'INITIAL_EVIDENCE': [initial_evidence]
    })
    prediction = pipeline.predict(user_data)
    return prediction[0]

def chat(inp, role="user", max_retries=3, initial_wait=1):
    message_history.append({"role": role, "content": f"{inp}"})
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message_history,
                max_tokens=200,
            )
            complition_history.append(completion)
            reply_content = completion.choices[0].message.content
            message_history.append({"role": "assistant", "content": f"{reply_content}"})
            return reply_content
        except (APITimeoutError, RateLimitError, InternalServerError,APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to get response from chat API after {max_retries} attempts: {str(e)}")
            wait_time = initial_wait * (2 ** attempt)
            print(f"API error occurred. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

@app.route('/')
def home():
    return jsonify({'error1':"hello smartmed"})
@app.route('/chat', methods=['POST'])
def chat_api():
    # user_input = request.args.get("user_input")
    user_input = request.json.get("user_input")
    content = request.json
    
    if user_input == "KINDLY PROVIDE THE OUTPUT":
        try:
            message_for_extraction.append(chat(user_input))
        
            # Extract evidence and make prediction
            text = message_for_extraction[-1]


        
            # Regular expression to match the evidences list
            evidences_pattern = r"evidences\s*=\s*\[(.*?)\]"
            evidences_match = re.search(evidences_pattern, text, re.DOTALL)

            # Regular expression to match the initial evidence
            initial_evidence_pattern = r"initial_evidence\s*=\s*\"(.*?)\""
            initial_evidence_match = re.search(initial_evidence_pattern, text)
            if evidences_match:
                # Extract the evidences list and clean it up
                evidences_str = evidences_match.group(1).strip()
                evidence = re.findall(r'"(.*?)"', evidences_str)
            else:
                evidence = []

            if initial_evidence_match:
                initial_evidence = initial_evidence_match.group(1)
            else:
                initial_evidence = None


            converted_initial_evidence = merged_evidences[initial_evidence][initial_evidence]

            symptoms = []
            for row in evidence:
                splitarray = row.split('_@_')
                if len(splitarray) == 2 and "V" in splitarray[1]:
                    Ev = splitarray[0]
                    Va = splitarray[1]
                    symptoms.append(f"{merged_evidences[Ev][Ev]}_@_{merged_evidences[Ev]['Values'][Va]}")
                elif len(splitarray) == 2:
                    Ev = splitarray[0]
                    symptoms.append(f"{merged_evidences[Ev][Ev]}_@_{splitarray[1]}")
                else:
                    Ev = splitarray[0]
                    symptoms.append(f"{merged_evidences[Ev][Ev]}")

            age = content.get('age', 25) 
            sex = content.get('sex', 'M')  
        
            predicted_pathology = predict_pathology(age, sex, symptoms, converted_initial_evidence)
        
            if predicted_pathology:
                return jsonify({
                    "predicted_pathology": f"Your predicted pathology is {merged_conditions[predicted_pathology]}",
                })
            else:
                return jsonify({
                    "output": message_for_extraction[-1],
                    "error": 'Unable to make prediction'
                }), 400
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    else:
        try:
            reply = chat(user_input)
            return jsonify({"user_input": user_input, "reply": reply})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/questions', methods=['GET'])
def get_questions():
    return jsonify([[data[i]['question_en'],i] for i in data.keys()])

@app.route('/answers', methods=['GET'])
def get_answers():
    question = request.args.get('question')
    if question not in data:
        return jsonify({'error': 'Question not found'}), 404
    
    data_type = data[question]['data_type']
    if data_type == 'M':
        return jsonify([[v['en'], k] for k, v in data[question]['value_meaning'].items()])
    elif data_type == 'B':
        return jsonify(['True', 'False'])
    else:
        return jsonify(data[question]['possible-values'])

@app.route('/initial_evidences', methods=['GET'])
def get_initial_evidences():
    return jsonify(list(merged_conditions.values()))

if __name__ == '__main__':
    app.run(debug=True)
# main_prompt = prompts.system_message
# message_history = [{"role": "user", "content": main_prompt}]
# completion_history = []
# message_for_extraction = []

# # Load data and models
# with open("release_evidences.json", "r") as file:
#     data = json.load(file)

# merged_evidences = json.load(open('merged_evidences.json'))
# merged_conditions = json.load(open('release_conditions.json'))

# class ExtendedLabelEncoder(LabelEncoder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def transform(self, y):
#         unseen_labels = set(y) - set(self.classes_)
#         if unseen_labels:
#             self.classes_ = np.append(self.classes_, list(unseen_labels))
#         return super().transform(y)

# def load_model_and_encoder():
#     try:
#         pipeline = joblib.load('pathology_model.pkl')
#         encoder_sex = joblib.load('encoder_sex.pkl')
#     except FileNotFoundError:
#         print("Model or encoder file not found. Please ensure the files are saved correctly.")
#         return None, None
#     return pipeline, encoder_sex

# def predict_pathology(age, sex, symptoms, initial_evidence):
#     pipeline, encoder_sex = load_model_and_encoder()
#     if not pipeline or not encoder_sex:
#         return None
#     try:
#         sex_encoded = encoder_sex.transform([sex])[0]
#     except ValueError as e:
#         print(f"Error encoding 'SEX': {e}")
#         sex_encoded = encoder_sex.transform(['unknown'])[0] 
    
#     user_data = pd.DataFrame({
#         'AGE': [age],
#         'SEX': [sex_encoded],
#         'DIFFERENTIAL_DIAGNOSIS': ['no_data'],
#         'EVIDENCES': [' '.join(symptoms) if symptoms else 'no_data'],
#         'INITIAL_EVIDENCE': [initial_evidence]
#     })
#     prediction = pipeline.predict(user_data)
#     return prediction[0]

# def chat(inp, role="user"):
#     message_history.append({"role": role, "content": f"{inp}"})
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=message_history,
#         max_tokens=200,
#     )
#     completion_history.append(completion)
#     reply_content = completion.choices[0].message.content
#     message_history.append({"role": "assistant", "content": f"{reply_content}"})
#     return reply_content

# @app.route('/')
# def home():
#     return jsonify({'error1':"hello"})
#     # user_input = request.args.get("user_input")
#     # if user_input == "KINDLY PROVIDE THE OUTPUT":
#     #     message_for_extraction.append(chat(user_input))
#     #     return jsonify({"output": message_for_extraction[-1]})
#     # else:
#     #     reply = chat(user_input)
#     #     return jsonify({"user_input": user_input, "reply": reply})


# @app.route('/chat', methods=['GET'])
# def chat_api():
#     # return jsonify({'error2':"hello"})
#     user_input = request.args.get("user_input")
#     if user_input == "KINDLY PROVIDE THE OUTPUT":
#         message_for_extraction.append(chat(user_input))
#         return jsonify({"output": message_for_extraction[-1]})
#     else:
#         reply = chat(user_input)
#         return jsonify({"user_input": user_input, "reply": reply})

# @app.route('/extract_evidence', methods=['POST'])
# def extract_evidence():
#     content = request.json
#     age = content.get('age', 25)  # Default to 50 if not provided
#     sex = content.get('sex', 'M')  # Default to 'M' if not provided
#     text = message_for_extraction[-1]

#     # Extract evidence
#     evidence_pattern = r'evidence = \[(.*?)\]'
#     evidence_matches = re.findall(evidence_pattern, text, re.DOTALL)
#     evidence = re.findall(r'"(E_\d+(_@_\d+|_@_V_\d+)?)"', evidence_matches[0])
#     evidence = [match[0] for match in evidence]

#     # Extract initial_evidence
#     initial_evidence_pattern = r'initial_evidence = "(E_\d+)"'
#     initial_evidence = re.findall(initial_evidence_pattern, text)[0]

#     converted_initial_evidence = merged_evidences[initial_evidence][initial_evidence]

#     symptoms = []
#     for row in evidence:
#         splitarray = row.split('_@_')
#         if len(splitarray) == 2 and "V" in splitarray[1]:
#             Ev = splitarray[0]
#             Va = splitarray[1]
#             symptoms.append(f"{merged_evidences[Ev][Ev]}_@_{merged_evidences[Ev]['Values'][Va]}")
#         elif len(splitarray) == 2:
#             Ev = splitarray[0]
#             symptoms.append(f"{merged_evidences[Ev][Ev]}_@_{splitarray[1]}")
#         else:
#             Ev = splitarray[0]
#             symptoms.append(f"{merged_evidences[Ev][Ev]}")

#     predicted_pathology = predict_pathology(age, sex, symptoms, converted_initial_evidence)
    
#     if predicted_pathology:
#         return jsonify({
#             "predicted_pathology": merged_conditions[predicted_pathology],
#             "age": age,
#             "sex": sex
#         })
#     else:
#         return jsonify({"error": 'Unable to make prediction'}), 400

# @app.route('/questions', methods=['GET'])
# def get_questions():
#     return jsonify([[data[i]['question_en'],i] for i in data.keys()])

# @app.route('/answers', methods=['GET'])
# def get_answers():
#     question = request.args.get('question')
#     if question not in data:
#         return jsonify({'error': 'Question not found'}), 404
    
#     data_type = data[question]['data_type']
#     if data_type == 'M':
#         return jsonify([[v['en'], k] for k, v in data[question]['value_meaning'].items()])
#     elif data_type == 'B':
#         return jsonify(['True', 'False'])
#     else:
#         return jsonify(data[question]['possible-values'])

# @app.route('/initial_evidences', methods=['GET'])
# def get_initial_evidences():
#     return jsonify(list(merged_conditions.values()))

# @app.route('/predict', methods=['POST'])
# def predict():
#     content = request.json
#     age = content.get('age', 25)  # Default to 25 if not provided
#     sex = content.get('sex', 'M')  # Default to 'M' if not provided
#     symptoms = content.get('symptoms', [])
#     initial_evidence = content.get('initial_evidence', 'no_data')

#     predicted_pathology = predict_pathology(age, sex, symptoms, initial_evidence)
    
#     if predicted_pathology:
#         return jsonify({
#             "predicted_pathology": merged_conditions[predicted_pathology],
#             "age": age,
#             "sex": sex,
#             "symptoms": symptoms,
#             "initial_evidence": initial_evidence
#         })
#     else:
#         return jsonify({"error": 'Unable to make prediction'}), 400

