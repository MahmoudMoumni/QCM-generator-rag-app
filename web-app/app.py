from flask import Flask, render_template, request, redirect, url_for, session,jsonify , Response
import uvicorn 
from asgiref.wsgi import WsgiToAsgi
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with secure key in production
from flask_cors import CORS
CORS(app)

import tempfile
import uuid

env_mode = os.getenv("ENVIRONMENT_MODE", "development")
if env_mode == "development":
    load_dotenv("./.env.development")


BACKEND_URL = os.getenv("BACKEND_URL")

# Directory for temporary search result files
TEMP_DIR = tempfile.gettempdir()


# --- Routes ---
import json
import requests

@app.route("/")
def home():
    session.clear()
    return render_template("home.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    session.clear()
    response = requests.get(f"{BACKEND_URL}/documents")
    if response.status_code == 200:
        print(response.text)
        json_response= json.loads(response.text)
        session["documents"] =json_response["documents"]
        print(session["documents"])
    else:
        session["documents"]=[]
    return render_template("index.html")
   

@app.route("/chatbot")
def chatbot():
    session.clear()
    return render_template("chatbot.html")

@app.route("/contextLens")
def contextLens():
    # Load results from the temporary file
    response = requests.get(f"{BACKEND_URL}/documents")
    if response.status_code == 200:
        #print(response.text)
        json_response= json.loads(response.text)
        session["documents"] =json_response["documents"]
        print(session["documents"])
        #print(session["documents"])
    else:
        session["documents"]=[]
    return render_template("contextLens.html")

@app.route("/search", methods=["POST"])
def search():
    search_results=[]
    user_input = request.json.get("user_input")
    response = requests.post(
        f"{BACKEND_URL}/search",
        json={"user_input": user_input},
        
    )
    if response.status_code == 200:
        json_response= json.loads(response.text)
        search_results =json_response["search_results"]
    else:
        search_results=[]
    return  jsonify({"search_results": search_results})


   
@app.route("/get-doc-url", methods=["POST"])
def get_doc_url():
    doc_url=""
    doc_id = request.json.get("doc_id")
    response = requests.post(
        f"{BACKEND_URL}/get-doc-url",
        json={"doc_id": doc_id},
        
    )
    if response.status_code == 200:
        json_response= json.loads(response.text)
        doc_url =json_response["doc_url"]
    
    return  jsonify({"doc_url": doc_url}) 


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")

    def generate():
        response = requests.post(
            f"{BACKEND_URL}/rag_stream",
            json={"user_input": user_input},
            stream=True
        )
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                yield chunk

    return Response(generate(), content_type='text/plain')

@app.route("/generate_quizz", methods=["POST"])
def generate_quizz():
    session.clear()
    session["show_congrats"]=False
    num_questions =int(request.form["num_questions"])
    question_type = request.form.get("question_type")
    keywords_json = request.form.get("keywords")
    keywords = json.loads(keywords_json) if keywords_json else []
    print("Received keywords:", keywords)
    selected_doc_id = request.form.get("file_name")
    print("selected documents ids")
    print(selected_doc_id)     
 
    # Encapsulate metadata in one object
    metadata = {
        "num_questions": num_questions,
        "question_type": question_type,
        "keywords":keywords,
        "selected_doc_id":selected_doc_id
    }

    data = {
        "metadata": json.dumps(metadata)  # Send it as a single field
    }
     
    response = requests.post(f"{BACKEND_URL}/generate_quizz", data=data)

    if response.status_code == 200:
        print(response.text)
        quizzes = json.loads(response.text)
        print(len(quizzes))
        session["quizzes"]=quizzes["quizzes"]
        return jsonify({"success": True, "redirect_url": url_for("quiz")})
    else:
        return jsonify({"success": False, "message": "an error happened we are sorry !"}), 400




@app.route('/upload_file', methods=['POST'])
def run_function():
    print(request)
    file = request.files.get("file")
    if file:
        file = request.files["file"]
        files = {
            "file": (file.filename, file.stream, file.mimetype)
        }
        response = requests.post(f"{BACKEND_URL}/upload_file", files=files)
        if response.status_code == 200:
            print(response.text)
            documents=json.loads(response.text)
            session["documents"] = documents
            return documents,200
        else:
            print(response.text)
            return [], 400


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    # Simulating a session holding questions
    print("quizzes page")
    quizzes=session.get('quizzes',[])
    feedback=session.get('feedback',{})
    score=session.get('score',0)
    answers = session.get('answers',{})
    done=session.get('done',False)
    print(session)
    if request.method == "POST":
        # Process answers
        for quizz in quizzes:
            quizz_id =int(quizz["id"])
            correct_answers_indexes = set(quizz["correct_answers_indexes"])
            correct_answers= set(quizz["options"][i] for i in correct_answers_indexes)
            selected_answers_indexes =[int (idx) for idx in request.form.getlist(f"q{quizz_id}")]
            print("selected_answers_indexes")
            print(selected_answers_indexes)
            selected_answers= set(quizz["options"][i] for i in selected_answers_indexes)
            answers[quizz_id] = selected_answers_indexes
            print(correct_answers_indexes)
            print(set(selected_answers_indexes))
            
            if set(selected_answers_indexes) == correct_answers_indexes:
                feedback[quizz_id] = (f"✅ Correct!", "success")
                score += 1
            else:
                feedback[quizz_id] = (
                    f"❌ Incorrect. Your answer(s): {', '.join(selected_answers)} but Correct answer(s): {', '.join(correct_answers)}",
                    "danger"
                )

        session["answers"] = answers
        session["score"] = score
        session["feedback"] = feedback
        session["done"] = True  
        if session["score"]==len(quizzes):
            session["show_congrats"]=True
        return redirect(url_for('quiz'))

    
    return render_template("quiz.html", quizzes=quizzes)#rest of variables will be used through session

@app.route("/results")
def results():
    score = session.get('score', 0)
    questions_count = len(session.get('questions', []))
    return render_template("results.html", score=score, questions_count=questions_count)


# Wrap the Flask app with ASGI adapter
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    uvicorn.run(asgi_app,host="0.0.0.0", port=5000)