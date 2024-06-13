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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the chunks using OpenAI embeddings and Chroma for vector storage
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,max_tokens=200)
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


# Streamlit app
def main():
    st.set_page_config(page_title="Job Scout for Data Science", page_icon=":mag:", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background-color: #f8f9fa;
            color: #343a41;
        }
        .stButton>button {
            color: white;
            background-color: #007bff;
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
            background-color: #0056b3;
        }
        
        .stTextArea textarea, .stTextInput input {
            background-color: #e9ecef;
            border: 1px solid #ced4da;
        }
        footer {
            font-size: 18px;
            text-align: center;
            padding: 10px;
            background-color: #007bff;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔍 Job Scout for Data Science")
    st.markdown("This interface is a tool designed to assist users in analyzing their resumes and job descriptions, providing personalized advice based on the content. Built on Retrieval-Augmented Generation (RAG), it leverages a combination of document retrieval and language model generation to offer contextual and informative replies. Users can input their resumes (text or PDF) and job descriptions, select from example questions, or type custom queries. Using LangChain, the application retrieves relevant information from a knowledge base consisting of data science interview prep material and curated resources stored in vector databases.")

    resume_text = st.text_area("Paste your resume text here", height=200)
    resume_file = st.file_uploader("Or upload your resume (PDF)", type="pdf")
    job_description = st.text_area("Paste the job description here", height=200)

    # Example question prompts
    st.markdown("##### Example Questions")
    st.markdown("Example 1: Am I a good fit for this job?")
    st.markdown("Example 2: Play the devil's advocate and identify what is missing in my resume for this role")
    st.markdown("Example 3: What are the key points to include in a cover letter?")
    
    question = st.text_input("Ask a question about the job or resume")

    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n")))
    
    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    template = """Answer the following question based on this context:{context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)
    final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser())

    if st.button("Submit"):
        if not resume_text and resume_file:
            resume_text = extract_text_from_pdf(resume_file)

        if resume_text and job_description:
            con="Based on my resume :"+resume_text + "\n\n" + "And the job description: "+job_description
            response = final_rag_chain.invoke({"question":con+", Answer this question in detail: "+question})
            st.write("Response:")
            st.write(response)
        else:
            st.write("Please provide either resume text or upload a resume PDF and also provide the job description.")
    st.markdown("##### Developed by Srikar Devulapalli")

if __name__ == "__main__":
    main()
