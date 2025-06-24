import os
import pickle
import faiss
import numpy as np
import smtplib
import time
import re
import threading
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq Model Configuration
GROQ_MODEL = "Llama3-70b-8192"
groq_client = Groq(api_key=GROQ_API_KEY)

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and documents
faiss_index_path = "faiss_index.bin"
documents_path = "documents.pkl"

if os.path.exists(faiss_index_path) and os.path.exists(documents_path):
    index = faiss.read_index(faiss_index_path)
    with open(documents_path, "rb") as f:
        data = pickle.load(f)
    documents = data["texts"]
else:
    raise Exception("FAISS index or documents file not found!")

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory storage for user conversations
conversations: Dict[str, List[Dict[str, str]]] = {}
user_sessions: Dict[str, Dict[str, str]] = {}  # Stores user email & name
last_activity: Dict[str, float] = {}  # Tracks last activity time

# Function to Send Emails
def send_email(subject, body, recipient_email):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, recipient_email, msg.as_string())
        print(f"✅ Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")

# Request Model for Chat
class UserInput(BaseModel):
    session_id: str
    message: str

# Function to limit token usage
def truncate_text(text, max_chars=800):
    return text[:max_chars] + "..." if len(text) > max_chars else text

# Function to detect if user wants to end chat
def user_wants_to_end_chat(message: str) -> bool:
    end_phrases = ["i am done", "i don't want to chat further", "end chat", "stop chat", "goodbye"]
    return any(phrase in message.lower() for phrase in end_phrases)

# Function to extract email from text
def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

# Function to handle auto-end chat
def auto_end_chat(session_id: str):
    time.sleep(120)  # Wait for 60 seconds
    if session_id in last_activity and time.time() - last_activity[session_id] > 120:
        end_chat(session_id)

@app.post("/chat/")
def chat_with_bot(input: UserInput):
    session_id = input.session_id
    user_query = truncate_text(input.message)  #Truncate long queries

    # Check if user wants to end chat
    if user_wants_to_end_chat(user_query):
        return end_chat(session_id)

    # Initialize user session if not exists
    if session_id not in user_sessions:
        user_sessions[session_id] = {"name": None, "email": None, "greeted": False}

    user_data = user_sessions[session_id]

    # Step 1: Greet the user if not greeted
    if not user_data["greeted"]:
        user_data["greeted"] = True
        return {"response": "Hello! 😊 Welcome to Prismatic Chatbot. What's your name?"}

    # Step 2: Collect the user's name
    if user_data["name"] is None:
        user_data["name"] = user_query.strip()
        return {"response": f"Nice to meet you, {user_data['name']}! Now, please enter your email."}

    #Step 3: Collect the user's email
    extracted_email = extract_email(user_query)
    if user_data["email"] is None:
        if extracted_email:
            user_data["email"] = extracted_email
            return {"response": f"Great! You're all set, {user_data['name']} 😊. How can I assist you today?"}
        else:
            return {"response": "Please enter a valid email to continue."}

    #Step 4: Proceed with normal chatbot logic after collecting name & email
    name = user_data["name"]
    email = user_data["email"]

    #Generate embeddings using Sentence Transformers
    query_embedding = np.array(embedding_model.encode([user_query], convert_to_numpy=True))

    #Retrieve the **top 2** most relevant documents
    _, indices = index.search(query_embedding, k=2)
    retrieved_texts = [truncate_text(documents[idx], max_chars=800) for idx in indices[0]]
    context = "\n".join(retrieved_texts)

    #Retrieve only the last 3 messages from conversation history
    conversation_history = conversations.get(session_id, [])[-3:]
    formatted_history = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in conversation_history])

    #Groq API Call
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": """You are Prismatic Chatbot, the virtual assistant for Prismatic Technologies.
                                                 Your role is to assist users with our services and frequently asked questions.
                                                 Always provide concise, correct, and clear answers.only answer the question relevant to the company.
                                                 Share links in that way is product or service name with link.Make sure provided the correct
                                                 and valid links from the given documts only.if client shows intrest in any product and asked further about the product then tell about the prouct 
                                                 features such how can help them.Try to keep the answer shorter as possible.if user asked about the what is 
                                                 your services then tell we provide the services relevavt the Shopify, wordpress, mobile developement,
                                                 Web developement, and marketing services. If a client asks to schedule a meeting
                                                 or demo then asked for his convinent time and schedule meeting according to his availbility
                                                 or wants to buy any service, then only provide contact details.Make sure dont share the 
                                                 contact ever time.
                                                 📞 +92-307-8881432
                                                 📧 info@prismatic-technologies.com
                                                 Head Office: 71-C3 Gulberg III, Lahore, Pakistan
                                                 Additional Office: Riyadh, Saudi Arabia
                                                 Office Hours: Monday to Friday, 9:00 AM to 6:00 PM
Employees or team information List:
1-Muhammad Mubashir our Full Stack Developer with skills as: ASP.NET, ASP.NET Core,Web API’s, SQL Server, Angular JS, ReactJS, JavaScript, AJAX, jQuery, Bootst>
2-Haseeb Nadeem our Front-end Developer with skills: HTML5, CSS3, Bootstrap5, JavaScript, Figma to HTML convert  3 Years
3-Bilal Javaid  our Front-end Developer with skills: HTML5, CSS3, Bootstrap5, JavaScript, Figma to HTML convert,WordPress Website Development,Shopify Store Des>
4-Muhammad Nouman our Business Analyst with skills: Project Scrum Planning, Implementation Head 4.5 years
5-ZAIN ALI our Full Stack Developer with skills: HTML5, CSS3, Bootstrap5, JavaScript,PHP Laravel, jQuery, MySQL, SQLSRV 5 Years 
6-Saima Shammas our Backend Developer with skills: HTML5, CSS3, Bootstrap5, JavaScript,PHP Laravel, jQuery, MySQL, SQLSRV 3 years
7-KASHIF ALI our CTO with skills: GitHub Pipeline, New technologies trends, technical analysis, Solution for the blockers, Laravel, Django, React, Vue, My Sql,>
8-M Athar Riaz  our Full Stack Developer with skills: PHP, Laravel, Vanilla JavaScript, Vue JS, React JS, jQuery, GitHub, Docker, SQL, NoSQL, HTML, Bootstrap, >
9-Sidra Nawazish our Front-End Developer with skills :HTML5, CSS3, Bootstrap5, JavaScript, Figma to HTML convert        3 Years 33303-8985935-4
10-Saeed ullah Saif our Full Stack Developer with skills: ASP.NET, ASP.NET Core, Web API’s,SQL Server, Angular JS, ReactJS, JavaScript, AJAX, jQuery, Bootstrap>
11-Ahtesham our .Net developer with skills: .net core, .net framework, angular, angular js      5 years
12-Mohsin Kamal our React Native Developer with skills: Vs Code, Android Studio, Java script, Type Script, Firebase, Apis       3 years 
13-Aqib Khan our Laravel Developer with skills: Laravel, Php,3rd party API’s, HTML, CSS 3 years 
14-Ali Abdullah our React Native Developer with skills: Vs Code, Android Studio, Java script, Type Script, Firebase, Apis       3 years 
15-Ayesha our SQA Engineer with skills: Manual testing, Automation testing, Database, API, Java core, Performance testing, Backend, UI/UX       3 years 
16-Muhammad Hamza our AI/Ml Engineer with skills: Ml, AI , GenAI and AI automation specialist with 2 two years of experties.
17-Muhammad Ahmad our DevOps Engineer with skills: Linux & Window Server Admin, CI/CD(Jenkins, GitHub Actions), Docker, Kubernetes, AWS 3 years 
18-Hamad Tariq our IT Infrastructure and DevOps Engineer with skills: Networking, System Administration, Cloud Support, IT Operations, OS Virtualization, ITIL >
19-Abdul Wahab Qamar our Front End Developer with skills: HTML5, CSS3, JavaScript, Vue.js, React.js and Rest Api’s Integration  3 Years
20-Maqsud Ur Rehman our Full Stack Developer with skills: C#, ASP.NET, ASP.NET Core,Web API’s,SQL Server, Angular JS, Angular TypeScript, JavaScript, AJAX,
21-Muhammad Safdar – Co-founder and Director
“Our business growth is based on the fact that you have an honest desire to help someone else.”
With this philosophy, Muhammad Safdar has been a cornerstone in establishing Prismatic’s strong foundation and commitment to excellence. Make sure 
only provide information that is relevant and present in the given documents.always behave like a perfect chat bot like chatgpt"""},
                {"role": "user", "content": f"Name: {name}, Email: {email}"},
                {"role": "user", "content": f"Previous Chats:\n{formatted_history}\n\nContext: {context}\n\nUser: {user_query}"}
            ],
            temperature=0.9,
            top_p=0.9,
            max_tokens=800
        )
        bot_reply = chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    #Store conversation in memory
    if session_id not in conversations:
        conversations[session_id] = []
    conversations[session_id].append({"user": user_query, "bot": bot_reply})

    #Update last activity time
    last_activity[session_id] = time.time()

    #Start auto-end chat thread
    threading.Thread(target=auto_end_chat, args=(session_id,), daemon=True).start()

    return {"response": bot_reply}

@app.get("/end_chat/{session_id}")
def end_chat(session_id: str):
    if session_id not in conversations:
        return {"message": "Session already ended or not found."}

    chat_history = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in conversations[session_id]])
    user_email = user_sessions.get(session_id, {}).get("email")

    if user_email:
        send_email("Your Chat Transcript with Prismatic Technologies Limited", chat_history, user_email)
    send_email("New User Chat Transcript", chat_history, EMAIL_USER)

    del conversations[session_id]
    del user_sessions[session_id]
    del last_activity[session_id]

    return {"message": "Chat transcript sent successfully!"}
