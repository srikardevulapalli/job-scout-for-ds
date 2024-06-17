# job-scout-for-ds

### Job Scout for Data Science :brain: :briefcase: :bar_chart:	

https://job-scout-for-data-science-fed899ff89d5.herokuapp.com/

###### Important Note!
###### _The current URL may stop working after the token limit is reached. Running the application on your local machine with your own API keys is recommended. Make sure to monitor your token usage and regenerate keys or update limits as needed._

Job Scout for Data Science is a tool designed to help users analyze their resumes and job descriptions, providing personalized advice using advanced machine learning techniques. This application leverages Retrieval-Augmented Generation (RAG) and Chain of Thought (CoT) reasoning to deliver contextually accurate and useful responses. 

#### Features
- **Resume & Job Description Analysis**: Input job descriptions and upload resumes (text or PDF) for tailored advice.
- **Personalized Advice**: Combines document retrieval and language model generation to provide detailed, context-rich responses.
- **Contextual Replies**: Utilizes Retrieval-Augmented Generation (RAG) and Chain of Thought (CoT) reasoning.
- **Custom Queries**: Select from example questions or type custom queries related to resumes, job descriptions, or general career advice.
- **Knowledge Base**: Utilizes a curated database of data science interview prep materials and resources.

#### Installation
Follow these steps to run the Streamlit app on your local machine:

##### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

##### Steps
1. Clone the Repository:
```
   git clone https://github.com/srikardevulapalli/job-scout-for-ds.git
   cd job-scout-for-ds
```
2. Create a Virtual Environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install Dependencies:
```
pip install -r requirements.txt
```
4. Set Environment Variables:
Create a .env file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```
5. Run the App:
```
streamlit run app.py
```

##### Additional Files
- Procfile: Used for deployment on Heroku.
- runtime.txt: Specifies the Python version.
- setup.sh: Shell script for setting up environment variables.
- requirements.txt: Lists the dependencies required for the app.

##### Usage
Once the app is running, you can:
- Input job descriptions and upload resumes to get personalized advice.
- Select from example questions or type custom queries.
- Receive detailed, context-rich responses to enhance your resume and job applications.

#### Customization:

- **Model and settings**: Swap GPT-3.5-turbo with another model, adjust the temperature for variability, and fine-tune token limits and text splitting for optimal performance.
- **Query Translation, Indexing & Retreival**: Can swap currently used Query decomposition, Chunking based Indexing & Chroma module for document retrieval.
- **Enhanced Knowledge Base**: Expand with additional data science resources and/or replace content with your domain’s. Can also integrate external APIs for comprehensive insights.
- **Flexible Interface**: Modify the UI in Streamlit with your choice.









