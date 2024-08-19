import streamlit as st
import os
from beyondllm import source, embeddings, retrieve, llms, generator
import config

# Loading API Keys from config file
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['HUGGINGFACE_ACCESS_TOKEN'] = os.environ['HF_TOKEN']

# Application title and introduction
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #0073e6;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .text-input {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #dddddd;
            margin-bottom: 20px;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
        }
        .button {
            display: inline-block;
            width: 100%;
            background-color: #0073e6;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #005bb5;
        }
        .result-section {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .highlight {
            background-color: #ffefc3;
            padding: 5px;
            border-radius: 5px;
            font-weight: bold;
        }
        .disabled-option {
            color: #d3d3d3;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>RAG Pipeline with BeyondLLM</h1>", unsafe_allow_html=True)

# Select data type from user (Only YouTube Video is selectable)
data_type = st.selectbox(
    "Select the type of data to process:", 
    [("YouTube Video", True), ("PDF (coming soon)", False), ("Web Page (coming soon)", False)], 
    format_func=lambda x: x[0]
)

# Get the data link from the user (only if YouTube Video is selected)
if data_type[1]:
    data_url = st.text_input("Enter the YouTube video URL:", "https://www.youtube.com/watch?v=ZM1bdh2mDJQ", key="data_url", label_visibility="collapsed")
    st.markdown("**Note:** Currently, only English YouTube videos are supported.")

# Data loading and embedding operations (only if YouTube Video is selected)
if st.button("Process Data", key="process_button") and data_type[1]:
    with st.spinner("Processing..."):
        data = source.fit(
            path=data_url,
            dtype="youtube",
            chunk_size=1024,
            chunk_overlap=0
        )
        
        model_name = 'BAAI/bge-small-en-v1.5'
        embed_model = embeddings.HuggingFaceEmbeddings(model_name=model_name)
        
        # Save retriever to session state
        st.session_state['retriever'] = retrieve.auto_retriever(
            data=data,
            embed_model=embed_model,
            type="cross-rerank",
            mode="OR",
            top_k=2
        )
        
        st.success("Data processed successfully!")

# Get query from user (only if YouTube Video is selected)
if data_type[1]:
    question = st.text_input("Enter your question:", key="question", label_visibility="collapsed")

# Show model and query results (only if YouTube Video is selected)
if st.button("Get Answer", key="answer_button") and data_type[1]:
    if 'retriever' in st.session_state:
        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
        
        llm = llms.HuggingFaceHubModel(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.environ.get('HF_TOKEN')
        )
        
        system_prompt = f"""
        <s>[INST]
        You are an AI Assistant.
        Please provide direct answers to questions.
        [/INST]
        </s>
        """
        
        pipeline = generator.Generate(
            question=question,
            retriever=st.session_state['retriever'],
            system_prompt=system_prompt,
            llm=llm
        )
        
        response = pipeline.call()
        st.markdown(f"<p><strong>Model response:</strong> <span class='highlight'>{response}</span></p>", unsafe_allow_html=True)
        
        # Displaying RAG Triad assessment results
        rag_evals = pipeline.get_rag_triad_evals()
        st.markdown(f"<p><strong>RAG Triad Evaluation:</strong> <span class='highlight'>{rag_evals}</span></p>", unsafe_allow_html=True)

        # Getting feedback from the user
        feedback = st.radio("Was this answer helpful?", ["Yes", "No"])
        if feedback == "No":
            feedback_comment = st.text_area("What was wrong with the answer? Please provide details.")
        else:
            st.success("Thank you for your feedback!")
    else:
        st.error("Please process the data before asking a question.")
