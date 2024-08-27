from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import joblib
import numpy as np
from openai import OpenAI
import prompts
import os
import time
from openai import APITimeoutError, RateLimitError, InternalServerError, APIConnectionError
import logging
from dotenv import load_dotenv

# Import your custom encoder
from custom_encoder import ExtendedLabelEncoder

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

def load_model_and_encoder():
    try:
        pipeline = joblib.load('pathology_model.pkl')
        encoder_sex = joblib.load('encoder_sex.pkl')
    except FileNotFoundError:
        print(f"AttributeError during loading: {e}")
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
        except (APITimeoutError, RateLimitError, InternalServerError, APIConnectionError) as e:
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
    user_input = request.json.get("user_input")
    content = request.json
    if user_input == "KINDLY PROVIDE THE OUTPUT":
        try:
            message_for_extraction.append(chat(user_input))
            text = message_for_extraction[-1]
            evidences_pattern = r"evidences\s*=\s*\[(.*?)\]"
            evidences_match = re.search(evidences_pattern, text, re.DOTALL)
            initial_evidence_pattern = r"initial_evidence\s*=\s*\"(.*?)\""
            initial_evidence_match = re.search(initial_evidence_pattern, text)

            if evidences_match:
                evidences_str = evidences_match.group(1).strip()
                evidence = re.findall(r'"(.*?)"', evidences_str)
            else:
                evidence = []

            if initial_evidence_match:
                initial_evidence = initial_evidence_match.group(1)
            else:
                initial_evidence = "E_56"

            # Handle the case where initial_evidence may not be found
            converted_initial_evidence = merged_evidences.get(initial_evidence, {}).get(initial_evidence, "Default_Evidence")

            symptoms = []
            for row in evidence:
                splitarray = row.split('_@_')
                if len(splitarray) == 2 and "V" in splitarray[1]:
                    Ev = splitarray[0]
                    Va = splitarray[1]
                    if merged_evidences.get(Ev, {}).get('Values', {}).get(Va):
                        symptoms.append(f"{merged_evidences[Ev].get(Ev, 'Unknown')}_@_{merged_evidences[Ev]['Values'][Va]}")
                elif len(splitarray) == 2:
                    Ev = splitarray[0]
                    symptoms.append(f"{merged_evidences.get(Ev, {}).get(Ev, 'Unknown')}_@_{splitarray[1]}")
                else:
                    Ev = splitarray[0]
                    symptoms.append(f"{merged_evidences.get(Ev, {}).get(Ev, 'Unknown')}")

            age = content.get('age', 25)
            sex = content.get('sex', 'M')
            predicted_pathology = predict_pathology(age, sex, symptoms, converted_initial_evidence)

            if predicted_pathology:
                return jsonify({
                    "predicted_pathology": f"Your predicted pathology is {merged_conditions.get(predicted_pathology, 'Unknown Pathology')}",
                })
            else:
                return jsonify({
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
    return jsonify([[data[i]['question_en'], i] for i in data.keys()])

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
