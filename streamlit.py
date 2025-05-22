import streamlit as st
import os
import re
import json
import hashlib
import shutil
from datetime import datetime

# Direct imports for AI functionality
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# ===== USER AUTHENTICATION SYSTEM =====
class UserSystem:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.base_pdf_dir = "user_data"
        self.load_users()
        self.ensure_directories()
    
    def ensure_directories(self):
        os.makedirs(self.base_pdf_dir, exist_ok=True)
    
    def get_user_pdf_dir(self, username):
        user_dir = os.path.join(self.base_pdf_dir, username, "pdfs")
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def get_user_vector_dir(self, username):
        vector_dir = os.path.join(self.base_pdf_dir, username, "vectors")
        os.makedirs(vector_dir, exist_ok=True)
        return vector_dir
    
    def load_users(self):
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        except:
            self.users = {}
    
    def save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, email=""):
        if username in self.users:
            return False, "Username already exists"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        self.users[username] = {
            "password": self.hash_password(password),
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "folders": [],
            "uploaded_pdfs": {}
        }
        
        self.get_user_pdf_dir(username)
        self.get_user_vector_dir(username)
        self.save_users()
        return True, "User registered successfully"
    
    def login_user(self, username, password):
        if username not in self.users:
            return False, "Username does not exist"
        
        if self.users[username]["password"] != self.hash_password(password):
            return False, "Incorrect password"
        
        self.users[username]["last_login"] = datetime.now().isoformat()
        self.save_users()
        return True, "Login successful"
    
    def get_user_data(self, username):
        return self.users.get(username, {})
    
    def update_user_folders(self, username, folders):
        if username in self.users:
            self.users[username]["folders"] = folders
            self.save_users()
    
    def save_user_pdf(self, username, uploaded_file, folder_id=None):
        try:
            user_pdf_dir = self.get_user_pdf_dir(username)
            
            base_name = uploaded_file.name
            file_path = os.path.join(user_pdf_dir, base_name)
            counter = 1
            
            while os.path.exists(file_path):
                name, ext = os.path.splitext(base_name)
                new_name = f"{name}_{counter}{ext}"
                file_path = os.path.join(user_pdf_dir, new_name)
                counter += 1
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            pdf_record = {
                "path": file_path,
                "original_name": uploaded_file.name,
                "upload_date": datetime.now().isoformat(),
                "folder_id": folder_id,
                "size": uploaded_file.size if hasattr(uploaded_file, 'size') else 0
            }
            
            self.users[username]["uploaded_pdfs"][os.path.basename(file_path)] = pdf_record
            self.save_users()
            
            return True, file_path
            
        except Exception as e:
            return False, f"Error saving PDF: {str(e)}"
    
    def get_user_pdfs(self, username, folder_id=None):
        user_data = self.get_user_data(username)
        pdfs = user_data.get("uploaded_pdfs", {})
        
        if folder_id is not None:
            filtered_pdfs = {}
            for filename, pdf_info in pdfs.items():
                if pdf_info.get("folder_id") == folder_id:
                    filtered_pdfs[filename] = pdf_info
            return filtered_pdfs
        
        return pdfs
    
    def delete_user_pdf(self, username, filename):
        try:
            user_data = self.get_user_data(username)
            pdfs = user_data.get("uploaded_pdfs", {})
            
            if filename in pdfs:
                file_path = pdfs[filename]["path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                user_vector_dir = self.get_user_vector_dir(username)
                vector_filename = os.path.splitext(filename)[0] + "_vectors"
                vector_path = os.path.join(user_vector_dir, vector_filename)
                if os.path.exists(vector_path):
                    shutil.rmtree(vector_path)
                
                del self.users[username]["uploaded_pdfs"][filename]
                self.save_users()
                
                return True, "PDF deleted successfully"
            else:
                return False, "PDF not found"
                
        except Exception as e:
            return False, f"Error deleting PDF: {str(e)}"

# ===== AI FUNCTIONALITY =====

@st.cache_resource
def init_ai_components():
    """Initialize AI model and embeddings"""
    try:
        model = Ollama(model="llama3.2")  # Change to your model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return model, embeddings
    except Exception as e:
        st.error(f"Failed to initialize AI components: {e}")
        return None, None

def create_vector_store_user(pdf_path, username):
    """Create vector store for user's PDF"""
    try:
        model, embeddings = init_ai_components()
        
        if embeddings is None:
            raise ValueError("Embeddings not initialized")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found in PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        db = FAISS.from_documents(texts, embeddings)
        return db
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e

def clean_quiz_output(raw_output):
    """Remove AI thinking process and return only clean questions"""
    text = str(raw_output)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if 'think' not in line.lower() or len(line.strip()) > 20:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def generate_quiz_user(db, quiz_type="MCQ", num_questions=5, difficulty="Medium", topic_focus=""):
    """Generate quiz questions from documents"""
    try:
        model, embeddings = init_ai_components()
        
        if model is None:
            return "Error: AI model not initialized"
        
        # Get relevant documents
        query = f"{difficulty} {quiz_type} quiz {num_questions} questions"
        if topic_focus:
            query += f" about {topic_focus}"
        
        docs = db.similarity_search(query, k=8)
        
        if not docs:
            return "No content available to generate quiz."
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if quiz_type == "MCQ":
            template = f"""
Create EXACTLY {num_questions} multiple choice questions based on the content below.

RULES:
- Do NOT show any thinking process
- Do NOT use <think> tags
- Output ONLY the questions in the exact format shown
- Make questions at {difficulty} difficulty level

FORMAT:
Question 1: [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: A

Question 2: [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: B

CONTENT: {{context}}

Generate {num_questions} questions now:
"""
        else:  # Open-ended
            template = f"""
Create EXACTLY {num_questions} open-ended questions based on the content below.

RULES:
- Do NOT show any thinking process
- Do NOT use <think> tags
- Output ONLY the questions
- Make questions at {difficulty} difficulty level

FORMAT:
Question 1: [Question text here]

Question 2: [Question text here]

Question 3: [Question text here]

CONTENT: {{context}}

Generate {num_questions} questions now:
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        result = chain.invoke({"context": context})
        clean_result = clean_quiz_output(str(result))
        
        return clean_result
        
    except Exception as e:
        return f"Error generating quiz: {str(e)}"

def tutor_response_user(question, db, mode="Explain"):
    """Generate tutor response based on documents"""
    try:
        model, embeddings = init_ai_components()
        
        if model is None:
            return "Error: AI model not initialized"
        
        docs = db.similarity_search(question, k=4)
        
        if not docs:
            return "No relevant information found in the PDF."
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if mode == "Quiz Me":
            template = """
You are providing feedback on a quiz answer. Be direct and helpful.

Question: {question}
Reference material: {context}

Provide clear feedback:
"""
        else:
            template = """
You are an AI tutor. Answer the student's question clearly and helpfully.

Question: {question}
Reference material: {context}

Provide a clear answer:
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        result = chain.invoke({"question": question, "context": context})
        clean_result = clean_quiz_output(str(result))
        
        return clean_result
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ===== QUIZ PARSING FUNCTIONS =====
def parse_mcq_questions(quiz_text):
    """Parse MCQ text into structured format"""
    quiz_text = clean_quiz_output(quiz_text)
    questions = []
    
    question_blocks = re.split(r'Question \d+:', quiz_text)
    question_blocks = [block.strip() for block in question_blocks if block.strip()]
    
    for i, block in enumerate(question_blocks):
        question_data = {}
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            continue
            
        question_text = lines[0].strip()
        
        if not question_text or 'think' in question_text.lower() or len(question_text) < 10:
            continue
            
        question_data['question'] = question_text
        question_data['number'] = len(questions) + 1
        question_data['options'] = {}
        question_data['correct_answer'] = None
        
        for line in lines[1:]:
            if line.startswith(('A)', 'B)', 'C)', 'D)')):
                option_letter = line[0]
                option_text = line[3:].strip()
                if option_text:
                    question_data['options'][option_letter] = option_text
            elif line.startswith('Correct Answer:'):
                question_data['correct_answer'] = line.split(':')[1].strip()
        
        if question_data['question'] and len(question_data['options']) >= 2:
            questions.append(question_data)
    
    return questions

def parse_open_ended_questions(quiz_text):
    """Parse open-ended questions into list"""
    quiz_text = clean_quiz_output(quiz_text)
    questions = []
    
    question_blocks = re.split(r'Question \d+:', quiz_text)
    question_blocks = [block.strip() for block in question_blocks if block.strip()]
    
    for i, block in enumerate(question_blocks):
        question_text = block.split('\n')[0].strip()
        
        if not question_text or 'think' in question_text.lower() or len(question_text) < 10:
            continue
            
        if question_text:
            questions.append({
                'number': len(questions) + 1,
                'question': question_text
            })
    
    return questions

def create_quiz_context_for_chatbot(questions, user_answers):
    """Create context string about current quiz for chatbot"""
    context_parts = []
    
    for q in questions:
        q_num = q['number']
        context_parts.append(f"Question {q_num}: {q['question']}")
        
        # Add options for MCQ
        if 'options' in q:
            for letter, text in q['options'].items():
                context_parts.append(f"  {letter}) {text}")
        
        # Add user's current answer if any
        if q_num in user_answers:
            context_parts.append(f"  Current answer: {user_answers[q_num]}")
        else:
            context_parts.append(f"  Current answer: Not answered yet")
        
        context_parts.append("")  # Empty line for separation
    
    return "\n".join(context_parts)

def create_open_quiz_context_for_chatbot(questions, user_answers):
    """Create context string about current open-ended quiz for chatbot"""
    context_parts = []
    
    for q in questions:
        q_num = q['number']
        context_parts.append(f"Question {q_num}: {q['question']}")
        
        # Add user's current answer if any
        if q_num in user_answers:
            context_parts.append(f"Current answer: {user_answers[q_num]}")
        else:
            context_parts.append(f"Current answer: Not answered yet")
        
        context_parts.append("")  # Empty line for separation
    
    return "\n".join(context_parts)

def display_mcq_quiz(questions):
    """Display interactive MCQ quiz with context-aware chatbot"""
    if not questions:
        st.error("No questions could be parsed from the quiz.")
        return None, None
    
    # Initialize chatbot state
    if 'quiz_chatbot_open' not in st.session_state:
        st.session_state.quiz_chatbot_open = False
    if 'quiz_chat_messages' not in st.session_state:
        st.session_state.quiz_chat_messages = []
    
    # Quiz header with owl chatbot icon
    col1, col2 = st.columns([10, 1])
    
    with col1:
        st.subheader(f"📝 MCQ Quiz ({len(questions)} Questions)")
    
    with col2:
        if st.button("🦉", help="Ask AI Tutor for help", key="quiz_chatbot_toggle"):
            st.session_state.quiz_chatbot_open = not st.session_state.quiz_chatbot_open
    
    # Create two columns - quiz on left, chatbot on right (when open)
    if st.session_state.quiz_chatbot_open:
        quiz_col, chat_col = st.columns([3, 2])
    else:
        quiz_col = st.container()
        chat_col = None
    
    user_answers = {}
    
    # Quiz content in left column
    with quiz_col:
        for q in questions:
            st.markdown("---")
            st.markdown(f"**Question {q['number']}:** {q['question']}")
            
            options_list = []
            options_display = []
            
            for letter, text in q['options'].items():
                options_list.append(letter)
                options_display.append(f"{letter}) {text}")
            
            if options_list:
                selected = st.radio(
                    f"Select your answer for Question {q['number']}:",
                    options_display,
                    key=f"mcq_q{q['number']}",
                    index=None
                )
                
                if selected:
                    user_answers[q['number']] = selected[0]
    
    # Context-aware chatbot in right column
    if st.session_state.quiz_chatbot_open and chat_col:
        with chat_col:
            st.markdown("### 🦉 AI Tutor")
            st.markdown("*Ask me about the quiz content or get help!*")
            
            # Create a container for chat messages with fixed height
            chat_container = st.container()
            
            with chat_container:
                # Display chat messages in a scrollable area
                if st.session_state.quiz_chat_messages:
                    for message in st.session_state.quiz_chat_messages:
                        if message["role"] == "user":
                            st.markdown(f"**You:** {message['content']}")
                        else:
                            st.markdown(f"**🦉 Tutor:** {message['content']}")
                        st.markdown("---")
                else:
                    st.info("👋 Hi! I'm your AI tutor. Ask me about any question or concept you're struggling with!")
            
            # Chat input
            chat_input = st.text_input(
                "Ask your question:",
                key="quiz_chat_input",
                placeholder="e.g., 'Can you explain question 2?' or 'What is customer registration?'"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                send_chat = st.button("Send", key="send_quiz_chat", type="primary")
            with col2:
                if st.button("Clear", key="clear_quiz_chat"):
                    st.session_state.quiz_chat_messages = []
                    st.rerun()
            
            if send_chat and chat_input.strip():
                # Add user message
                st.session_state.quiz_chat_messages.append({
                    "role": "user", 
                    "content": chat_input.strip()
                })
                
                # Generate context-aware response
                with st.spinner("🤔 Thinking..."):
                    # Create context from current quiz
                    quiz_context = create_quiz_context_for_chatbot(questions, user_answers)
                    enhanced_question = f"""
                    Context: The user is currently taking a quiz. Here are the quiz questions and their current answers:
                    
                    {quiz_context}
                    
                    User's question: {chat_input.strip()}
                    
                    Please provide helpful tutoring assistance. If they're asking about a specific question, refer to it by number.
                    """
                    
                    response = tutor_response_user(enhanced_question, st.session_state.db, "Explain")
                
                # Add AI response
                st.session_state.quiz_chat_messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # Clear input and refresh
                st.rerun()
            
            # Quick help buttons
            st.markdown("**Quick Help:**")
            help_col1, help_col2 = st.columns(2)
            
            with help_col1:
                if st.button("💡 Hint", key="quiz_hint", use_container_width=True):
                    if user_answers:
                        # Get hint for the last answered question
                        last_q_num = max(user_answers.keys())
                        last_question = next(q for q in questions if q['number'] == last_q_num)
                        hint_request = f"Give me a hint for this question without revealing the answer: {last_question['question']}"
                        
                        with st.spinner("💭 Generating hint..."):
                            hint_response = tutor_response_user(hint_request, st.session_state.db, "Explain")
                        
                        st.session_state.quiz_chat_messages.append({
                            "role": "user", 
                            "content": f"Hint for Question {last_q_num}"
                        })
                        st.session_state.quiz_chat_messages.append({
                            "role": "assistant", 
                            "content": hint_response
                        })
                        st.rerun()
            
            with help_col2:
                if st.button("📚 Explain", key="quiz_explain", use_container_width=True):
                    explain_request = "Can you explain the main concepts covered in this quiz?"
                    
                    with st.spinner("📖 Explaining concepts..."):
                        explain_response = tutor_response_user(explain_request, st.session_state.db, "Explain")
                    
                    st.session_state.quiz_chat_messages.append({
                        "role": "user", 
                        "content": "Explain main concepts"
                    })
                    st.session_state.quiz_chat_messages.append({
                        "role": "assistant", 
                        "content": explain_response
                    })
                    st.rerun()
    
    return user_answers, questions

def display_open_ended_quiz(questions):
    """Display open-ended quiz with context-aware chatbot"""
    if not questions:
        st.error("No questions could be parsed from the quiz.")
        return None, None
    
    # Initialize chatbot state for open-ended quiz
    if 'open_quiz_chatbot_open' not in st.session_state:
        st.session_state.open_quiz_chatbot_open = False
    if 'open_quiz_chat_messages' not in st.session_state:
        st.session_state.open_quiz_chat_messages = []
    
    # Quiz header with owl chatbot icon
    col1, col2 = st.columns([10, 1])
    
    with col1:
        st.subheader(f"📝 Open-ended Quiz ({len(questions)} Questions)")
    
    with col2:
        if st.button("🦉", help="Ask AI Tutor for help", key="open_quiz_chatbot_toggle"):
            st.session_state.open_quiz_chatbot_open = not st.session_state.open_quiz_chatbot_open
    
    # Create two columns - quiz on left, chatbot on right (when open)
    if st.session_state.open_quiz_chatbot_open:
        quiz_col, chat_col = st.columns([3, 2])
    else:
        quiz_col = st.container()
        chat_col = None
    
    user_answers = {}
    
    # Quiz content in left column
    with quiz_col:
        for q in questions:
            st.markdown("---")
            st.markdown(f"### Question {q['number']}")
            st.markdown(f"**{q['question']}**")
            
            answer = st.text_area(
                f"Your answer:",
                key=f"open_q{q['number']}",
                height=150,
                placeholder="Write your detailed answer here..."
            )
            
            if answer.strip():
                user_answers[q['number']] = answer.strip()
    
    # Context-aware chatbot for open-ended quiz
    if st.session_state.open_quiz_chatbot_open and chat_col:
        with chat_col:
            st.markdown("### 🦉 AI Tutor")
            st.markdown("*Get help with your answers!*")
            
            # Display chat messages
            if st.session_state.open_quiz_chat_messages:
                for message in st.session_state.open_quiz_chat_messages:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**🦉 Tutor:** {message['content']}")
                    st.markdown("---")
            else:
                st.info("👋 Hi! I can help you brainstorm ideas or check your answers!")
            
            # Chat input
            chat_input = st.text_input(
                "Ask your question:",
                key="open_quiz_chat_input",
                placeholder="e.g., 'Help me with question 1' or 'Is my answer correct?'"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                send_chat = st.button("Send", key="send_open_quiz_chat", type="primary")
            with col2:
                if st.button("Clear", key="clear_open_quiz_chat"):
                    st.session_state.open_quiz_chat_messages = []
                    st.rerun()
            
            if send_chat and chat_input.strip():
                # Add user message
                st.session_state.open_quiz_chat_messages.append({
                    "role": "user", 
                    "content": chat_input.strip()
                })
                
                # Generate context-aware response
                with st.spinner("🤔 Thinking..."):
                    # Create context from current quiz
                    quiz_context = create_open_quiz_context_for_chatbot(questions, user_answers)
                    enhanced_question = f"""
                    Context: The user is working on an open-ended quiz. Here are the questions and their current answers:
                    
                    {quiz_context}
                    
                    User's question: {chat_input.strip()}
                    
                    Please provide helpful tutoring assistance for their open-ended answers.
                    """
                    
                    response = tutor_response_user(enhanced_question, st.session_state.db, "Explain")
                
                # Add AI response
                st.session_state.open_quiz_chat_messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                st.rerun()
            
            # Quick help buttons for open-ended
            st.markdown("**Quick Help:**")
            help_col1, help_col2 = st.columns(2)
            
            with help_col1:
                if st.button("💭 Brainstorm", key="open_quiz_brainstorm", use_container_width=True):
                    if questions:
                        first_question = questions[0]['question']
                        brainstorm_request = f"Help me brainstorm ideas for this question: {first_question}"
                        
                        with st.spinner("💭 Brainstorming..."):
                            brainstorm_response = tutor_response_user(brainstorm_request, st.session_state.db, "Explain")
                        
                        st.session_state.open_quiz_chat_messages.append({
                            "role": "user", 
                            "content": "Help me brainstorm ideas"
                        })
                        st.session_state.open_quiz_chat_messages.append({
                            "role": "assistant", 
                            "content": brainstorm_response
                        })
                        st.rerun()
            
            with help_col2:
                if st.button("✍️ Review", key="open_quiz_review", use_container_width=True):
                    if user_answers:
                        last_q_num = max(user_answers.keys())
                        last_answer = user_answers[last_q_num]
                        last_question = next(q for q in questions if q['number'] == last_q_num)
                        
                        review_request = f"Please review my answer for '{last_question['question']}': {last_answer}"
                        
                        with st.spinner("✍️ Reviewing..."):
                            review_response = tutor_response_user(review_request, st.session_state.db, "Quiz Me")
                        
                        st.session_state.open_quiz_chat_messages.append({
                            "role": "user", 
                            "content": f"Review my answer for Question {last_q_num}"
                        })
                        st.session_state.open_quiz_chat_messages.append({
                            "role": "assistant", 
                            "content": review_response
                        })
                        st.rerun()
    
    return user_answers, questions

def check_mcq_answers(user_answers, questions):
    """Check MCQ answers and provide feedback"""
    if not user_answers:
        return "Please answer at least one question."
    
    correct_count = 0
    total_questions = len(questions)
    feedback = []
    
    for q in questions:
        q_num = q['number']
        correct_answer = q['correct_answer']
        user_answer = user_answers.get(q_num, "Not answered")
        
        if user_answer == correct_answer:
            correct_count += 1
            feedback.append(f"✅ **Question {q_num}**: Correct! ({correct_answer})")
        else:
            feedback.append(f"❌ **Question {q_num}**: Wrong. You answered {user_answer}, correct answer is {correct_answer}")
            feedback.append(f"   *{q['options'].get(correct_answer, 'N/A')}*")
    
    score = f"**Score: {correct_count}/{total_questions} ({correct_count/total_questions*100:.1f}%)**"
    
    return score + "\n\n" + "\n\n".join(feedback)

# ===== MAIN APP FUNCTIONS =====
def show_login_page():
    """Show login/register page"""
    st.markdown("# 🦉 AI Buddy ")
    st.markdown("Welcome! Please login or create an account to continue.")
    
    if 'user_system' not in st.session_state:
        st.session_state.user_system = UserSystem()
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Welcome Back!")
        
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🚀 Login", type="primary", use_container_width=True):
            if login_username and login_password:
                success, message = st.session_state.user_system.login_user(login_username, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    
                    user_data = st.session_state.user_system.get_user_data(login_username)
                    st.session_state.folders = user_data.get("folders", [])
                    
                    st.success(f"Welcome back, {login_username}! 🎉")
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please fill in all fields")
    
    with tab2:
        st.subheader("Create Your Account")
        
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("📝 Register", type="primary", use_container_width=True):
            if reg_username and reg_password and reg_confirm:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                else:
                    success, message = st.session_state.user_system.register_user(
                        reg_username, reg_password, reg_email
                    )
                    if success:
                        st.success(message)
                        st.info("✅ You can now login!")
                    else:
                        st.error(message)
            else:
                st.warning("Please fill in all required fields")

def handle_pdf_upload_user(uploaded_file, folder_id):
    """Handle PDF upload for user"""
    file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
    
    if file_key in st.session_state:
        return True, "PDF already processed!"
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Saving PDF file...")
        progress_bar.progress(25)
        
        success, file_path = st.session_state.user_system.save_user_pdf(
            st.session_state.username, uploaded_file, folder_id
        )
        
        if not success:
            return False, file_path
        
        status_text.text("🔄 Creating AI embeddings...")
        progress_bar.progress(50)
        
        db = create_vector_store_user(file_path, st.session_state.username)
        
        status_text.text("🔄 Saving vector database...")
        progress_bar.progress(75)
        
        user_vector_dir = st.session_state.user_system.get_user_vector_dir(st.session_state.username)
        vector_filename = os.path.splitext(os.path.basename(file_path))[0] + "_vectors"
        vector_path = os.path.join(user_vector_dir, vector_filename)
        
        db.save_local(vector_path)
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        st.session_state[file_key] = True
        
        return True, "PDF uploaded and processed successfully!"
        
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"

def show_main_app():
    """Show main application"""
    # Initialize folders
    if 'folders' not in st.session_state:
        user_data = st.session_state.user_system.get_user_data(st.session_state.username)
        st.session_state.folders = user_data.get("folders", [])
    
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = None
    if 'rename_mode' not in st.session_state:
        st.session_state.rename_mode = None
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🦉 AI Buddy ")
    
    with col2:
        user_pdfs = st.session_state.user_system.get_user_pdfs(st.session_state.username)
        st.metric("📄 Total PDFs", len(user_pdfs))
    
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            for key in ['logged_in', 'username', 'folders', 'selected_folder', 'app_mode']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Sidebar - Folder Management
    with st.sidebar:
        st.header("📁 Folders")
        
        # Add new folder button
        if st.button("➕ Add New Folder", type="primary", use_container_width=True):
            folder_name = f"New Folder {len(st.session_state.folders) + 1}"
            new_folder = {
                'id': len(st.session_state.folders),
                'name': folder_name,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.folders.append(new_folder)
            st.session_state.user_system.update_user_folders(st.session_state.username, st.session_state.folders)
            st.rerun()
        
        # Display folders
        if st.session_state.folders:
            st.markdown("---")
            
            for folder in st.session_state.folders:
                folder_id = folder['id']
                folder_name = folder['name']
                
                # Create columns for folder item
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Folder selection button
                    if st.button(
                        f"📂 {folder_name}", 
                        key=f"folder_{folder_id}",
                        use_container_width=True,
                        type="secondary" if st.session_state.selected_folder != folder_id else "primary"
                    ):
                        st.session_state.selected_folder = folder_id
                        st.session_state.app_mode = None  # Reset mode when switching folders
                        st.rerun()
                
                with col2:
                    # Rename button
                    if st.button("✏️", key=f"rename_{folder_id}", help="Rename folder"):
                        st.session_state.rename_mode = folder_id
                        st.rerun()
                
                with col3:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{folder_id}", help="Delete folder"):
                        # Delete folder and its PDFs
                        folder_pdfs = st.session_state.user_system.get_user_pdfs(st.session_state.username, folder_id)
                        for filename in folder_pdfs.keys():
                            st.session_state.user_system.delete_user_pdf(st.session_state.username, filename)
                        
                        st.session_state.folders = [f for f in st.session_state.folders if f['id'] != folder_id]
                        st.session_state.user_system.update_user_folders(st.session_state.username, st.session_state.folders)
                        
                        if st.session_state.selected_folder == folder_id:
                            st.session_state.selected_folder = None
                            st.session_state.app_mode = None
                        
                        st.rerun()
            
            # Rename dialog
            if st.session_state.rename_mode is not None:
                st.markdown("---")
                st.subheader("✏️ Rename Folder")
                
                current_folder = next(f for f in st.session_state.folders if f['id'] == st.session_state.rename_mode)
                new_name = st.text_input("New folder name:", value=current_folder['name'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Save", type="primary"):
                        if new_name.strip():
                            for folder in st.session_state.folders:
                                if folder['id'] == st.session_state.rename_mode:
                                    folder['name'] = new_name.strip()
                                    break
                            st.session_state.user_system.update_user_folders(st.session_state.username, st.session_state.folders)
                            st.session_state.rename_mode = None
                            st.rerun()
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.rename_mode = None
                        st.rerun()
        
        else:
            st.info("Click ➕ to create your first folder!")
    
    # Main Content Area
    if st.session_state.selected_folder is None:
        # No folder selected
        st.markdown("""
        ## 👋 Welcome to AI Buddy!
        
        **Get started in 3 easy steps:**
        
        1. **📁 Create a folder** - Click "➕ Add New Folder" in the sidebar
        2. **🎯 Choose your mode** - Quiz Generation or Chat with PDF  
        3. **📄 Upload your PDF** - Start learning immediately!
        
        ### 🚀 Features:
        - **Interactive MCQ Quizzes** with clickable answers
        - **Open-ended Questions** with AI feedback
        - **Chat with PDF** for instant Q&A
        - **Organized Folders** to manage your studies
        - **🦉 AI Tutor** - Context-aware chatbot during quizzes
        """)

    elif st.session_state.app_mode is None:
        # Folder selected, choose mode
        selected_folder = next(f for f in st.session_state.folders if f['id'] == st.session_state.selected_folder)
        
        st.header(f"📂 {selected_folder['name']}")
        st.subheader("🎯 Choose Your Learning Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Quiz Generator", type="primary", use_container_width=True):
                st.session_state.app_mode = "quiz"
                st.rerun()
            st.markdown("""
            **Perfect for:**
            - Testing your knowledge
            - Exam preparation  
            - Quick assessments
            - MCQ & Open-ended questions
            - **🦉 AI Tutor help during quiz**
            """)
        
        with col2:
            if st.button("💬 Chat with PDF", type="primary", use_container_width=True):
                st.session_state.app_mode = "chat"
                st.rerun()
            st.markdown("""
            **Perfect for:**
            - Understanding concepts
            - Getting explanations
            - Interactive learning
            - Q&A sessions
            """)

    else:
        # Mode selected, show the app functionality
        selected_folder = next(f for f in st.session_state.folders if f['id'] == st.session_state.selected_folder)
        
        # Back button
        if st.button("← Back to Mode Selection"):
            st.session_state.app_mode = None
            st.rerun()
        
        st.header(f"📂 {selected_folder['name']} - {st.session_state.app_mode.title()} Mode")
        
        # Show existing PDFs first
        user_pdfs = st.session_state.user_system.get_user_pdfs(st.session_state.username, st.session_state.selected_folder)
        
        if user_pdfs:
            st.subheader("📚 Your PDFs in this folder:")
            
            for filename, pdf_info in user_pdfs.items():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"📄 **{pdf_info['original_name']}**")
                
                with col2:
                    if st.button("🎯 Use this PDF", key=f"use_{filename}", use_container_width=True):
                        try:
                            user_vector_dir = st.session_state.user_system.get_user_vector_dir(st.session_state.username)
                            vector_filename = os.path.splitext(filename)[0] + "_vectors"
                            vector_path = os.path.join(user_vector_dir, vector_filename)
                            
                            if os.path.exists(vector_path):
                                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                                db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
                                
                                st.session_state.db = db
                                st.session_state.file_processed = True
                                st.session_state.current_pdf_name = pdf_info['original_name']
                                
                                st.success(f"✅ Ready to work with: **{pdf_info['original_name']}**")
                            else:
                                st.error("Please re-upload this PDF")
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                with col3:
                    if st.button("🗑️", key=f"delete_{filename}", help="Delete PDF"):
                        success, message = st.session_state.user_system.delete_user_pdf(st.session_state.username, filename)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            st.markdown("---")
        
        # File upload
        st.subheader("📁 Upload New PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file:",
            type="pdf",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
            
            if file_key not in st.session_state:
                with st.spinner("📚 Processing PDF..."):
                    success, message = handle_pdf_upload_user(uploaded_file, st.session_state.selected_folder)
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.info("✅ This PDF has already been processed!")
        
        # Show functionality if PDF is loaded
        if hasattr(st.session_state, 'file_processed') and st.session_state.file_processed:
            st.markdown("---")
            
            if st.session_state.app_mode == "quiz":
                # Quiz Generation Mode
                st.subheader(f"📝 Generate Quiz from: {st.session_state.current_pdf_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    quiz_type = st.selectbox("Select Quiz Type:", ["MCQ", "Open-ended"])
                    num_questions = st.slider("Number of Questions:", 1, 10, 3)
                
                with col2:
                    difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"])
                    topic_focus = st.text_input("Focus on specific topic (optional):")
                
                if st.button("🎯 Generate Quiz", type="primary"):
                    with st.spinner("🎲 Creating your quiz..."):
                        quiz_content = generate_quiz_user(st.session_state.db, quiz_type, num_questions, difficulty, topic_focus)
                        
                        st.session_state.current_quiz = quiz_content
                        st.session_state.quiz_type = quiz_type
                        st.session_state.quiz_generated = True
                
                # Display generated quiz
                if hasattr(st.session_state, 'quiz_generated') and st.session_state.quiz_generated:
                    st.markdown("---")
                    
                    if st.session_state.quiz_type == "MCQ":
                        questions = parse_mcq_questions(st.session_state.current_quiz)
                        user_answers, parsed_questions = display_mcq_quiz(questions)
                        
                        if st.button("✅ Check Answers", type="primary"):
                            if user_answers:
                                feedback = check_mcq_answers(user_answers, parsed_questions)
                                st.success("📊 Results:")
                                st.markdown(feedback)
                                
                                # Show AI tutor explanation for wrong answer
                                st.markdown("---")
                                st.subheader("🦉 AI Tutor Explanations for Incorrect Answers")
                                
                                for q in parsed_questions:
                                    q_num = q['number']
                                    correct_answer = q['correct_answer']
                                    user_answer = user_answers.get(q_num, "Not answered")
                    
                                    if user_answer != correct_answer and user_answer != "Not answered":
                                     with st.expander(f"🦉 Explanation for Question {q_num}"):
                                        with st.spinner("Generating explanation..."):
                                            explanation = tutor_response_user(
                                                f"Explain why '{correct_answer}) {q['options'].get(correct_answer, 'N/A')}' is the correct answer for: {q['question']}. The student chose '{user_answer}) {q['options'].get(user_answer, 'N/A')}'",
                                                st.session_state.db,
                                                "Explain"
                                            )
                                            st.write(explanation)
                            else:
                                st.warning("Please answer at least one question!")
                    
                    else:  # Open-ended
                        questions = parse_open_ended_questions(st.session_state.current_quiz)
                        user_answers, parsed_questions = display_open_ended_quiz(questions)
                        
                        if st.button("✅ Get Feedback", type="primary"):
                            if user_answers:
                                st.success("📊 Feedback:")
                                
                                for q_num, answer in user_answers.items():
                                    question_text = next(q['question'] for q in parsed_questions if q['number'] == q_num)
                                    
                                    with st.spinner(f"Evaluating Question {q_num}..."):
                                        feedback = tutor_response_user(f"Evaluate this answer for '{question_text}': {answer}", st.session_state.db, "Quiz Me")
                                        
                                        st.markdown(f"**Question {q_num} Feedback:**")
                                        st.write(feedback)
                            else:
                                st.warning("Please answer at least one question!")
            
            elif st.session_state.app_mode == "chat":
                # Chat Mode
                st.subheader(f"💬 Chat with: {st.session_state.current_pdf_name}")
                
                # Initialize chat history
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []
                
                # Display chat history
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask me anything about the PDF..."):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    with st.spinner("🤔 Thinking..."):
                        answer = tutor_response_user(prompt, st.session_state.db, "Explain")
                        
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.write(answer)
        
        else:
            st.info("👆 Please upload a PDF file or select an existing one to get started!")

# ===== MAIN APPLICATION =====
def main():
    """Main application"""
    st.set_page_config(
        page_title="AI Buddy", 
        page_icon="🦉", 
        layout="wide"
    )
    
    # Check authentication
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()