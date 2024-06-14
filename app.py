import os
import streamlit as st
from openai import OpenAI
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from PyPDF2 import PdfReader
from operator import itemgetter
import time

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")


os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Initialize OpenAI client
# client = OpenAI(organization='org-Vmf8l03IqwdFZNUKdsQK4n9j')

#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=(
        "https://www.dataquest.io/blog/how-data-science-resume-cv/",
        "https://365datascience.com/career-advice/job-interview-tips/data-science-resume/",
        "https://www.datacamp.com/blog/tips-to-build-your-data-scientist-resume",
        "https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:5fa21b89-07b8-3397-a735-dd18f2e49bdd",
        'http://hits.dwyl.io/%7Brbhatia46%7D/%7BData-Science-Interview-Resources%7D', 
        "https://towardsdatascience.com/data-science-interview-guide-4ee9f5dc778",
        "https://www.projectpro.io/article/100-data-science-interview-questions-and-answers-for-2021/184",
        "https://towardsdatascience.com/how-to-prepare-for-machine-learning-interviews-5fac3db58168",
        "https://towardsdatascience.com/red-flags-in-data-science-interviews-4f492bbed4c4",
        "https://www.usebraintrust.com/hire/interview-questions/generative-ai-specialists",
        "https://medium.com/predict/five-interview-questions-to-predict-a-good-data-scientist-40d310cdcd68",
        "https://towardsdatascience.com/my-take-on-data-scientist-interview-questions-part-1-6df22252b2e8",
        "https://medium.com/@jasonkgoodman/advice-on-building-data-portfolio-projects-c5f96d8a0627",
        "https://towardsdatascience.com/up-level-your-data-science-resume-getting-past-ats-64322f0cbb73",
        "https://www.freecodecamp.org/news/how-to-write-a-resume-that-works/",
        ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the chunks using OpenAI embeddings and Chroma for vector storage
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm= ChatAnthropic(
#     model="claude-3-sonnet-20240229",
#     temperature=0,
#     max_tokens=150)

# Post-processing to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def retrieve_and_rag(question,prompt,sub_question_generator_chain):
    """RAG on each sub-question"""
    
    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
    return rag_results,sub_questions

# Streamlit app
def main():
    st.set_page_config(page_title="Job Scout for Data Science", page_icon=":mag:", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300&family=Poppins:wght@500&display=swap');
        
        .main {
            background-image: url('https://img.freepik.com/free-photo/futuristic-sci-fi-space-tunnel-passageway-with-glowing-shiny-lights_181624-17286.jpg?t=st=1718338366~exp=1718341966~hmac=b404f9bb6e0b81ab5375cad5f7b7b9aaf6a42751c16e912f7a558c7b3dbe42a7&w=1380');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #FFFFFF;
        }
        .stButton>button {
            color: white;
            background-color: #17a2b8;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #138496;
        }
        
        .stTextArea textarea, .stTextInput input {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
        }
        
        .stTextArea label, .stTextInput label {
            color: #FFFFFF;
        }
        
        footer {
            font-size: 18px;
            text-align: center;
            padding: 10px;
            background-color: #17a2b8;
            color: white;
        }
        
        .title {
            font-family: 'Poppins', sans-serif;
            font-size: 48px;
            font-weight: 500;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">🔍 Job Scout for Data Science</div>', unsafe_allow_html=True)

    st.markdown("This interface is a tool designed to assist users in analyzing their resumes and job descriptions, providing personalized advice based on the content. Built on Retrieval-Augmented Generation (RAG), it leverages a combination of document retrieval and language model generation to offer contextual and informative replies. Users can input their resumes (text or PDF) and job descriptions, select from example questions, or type custom queries. Using LangChain, the application retrieves relevant information from a knowledge base consisting of data science interview prep material and curated resources stored in vector databases.")

    resume_text = st.text_area("Paste your resume text here", height=200)
    resume_file = st.file_uploader("Or upload your resume (PDF)", type="pdf")
    job_description = st.text_area("Paste the job description here", height=200)

    # Example question prompts
    st.markdown("##### Example Questions")
    st.markdown("Example 1: Am I a good fit for this job?")
    st.markdown("Example 2: Play the devil's advocate and identify what is missing in my resume for this role")
    st.markdown("Example 3: What are the key points to include in a cover letter?")
    
    question = st.text_input("Ask a question about the job,resume or anything in general")

    if not resume_text and resume_file:
      resume_text = extract_text_from_pdf(resume_file)
    if resume_text and job_description:
      con="Based on my resume :"+resume_text + "\n\n" + "And the job description: "+job_description
      question=con+", Answer this question in detail: "+question
    else:
      question="Answer this question in detail: "+question
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    generate_queries_decomposition = ( prompt_decomposition | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n")))
    questions = generate_queries_decomposition.invoke({"question":question})
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)
    q_a_pairs = ""
    for q in questions:
      rag_chain = (
      {"context": itemgetter("question") | retriever, 
      "question": itemgetter("question"),
      "q_a_pairs": itemgetter("q_a_pairs")} 
      | decomposition_prompt
      | llm
      | StrOutputParser())

      answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
      q_a_pair = format_qa_pair(q,answer)
      q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    prompt = hub.pull("rlm/rag-prompt")
    answers, questions = retrieve_and_rag(question, prompt, generate_queries_decomposition)
    def format_qa_pairs(questions, answers):
      """Format Q and A pairs"""
    
      formatted_string = ""
      for i, (question, answer) in enumerate(zip(questions, answers), start=1):
          formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
      return formatted_string.strip()
    
    context = format_qa_pairs(questions, answers)
    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
         
   
    if st.button("Submit"):
        with st.spinner('Processing...'):
            time.sleep(3)  # simulate a delay
            response = final_rag_chain.invoke({"context": context, "question": question})
            
        st.write("Response:")
        st.write(response)

    st.markdown("##### Developed by Srikar Devulapalli")

if __name__ == "__main__":
    main()
