import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from config import model_name, loaded_model, members, markdown_text


tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = T5ForConditionalGeneration.from_pretrained(loaded_model, device_map='auto', torch_dtype=torch.float32 )

#file loader and preprocessing  
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts


#LLM pipeline
def llm_pipeline(input_data, max_length = 500, min_length = 50):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = max_length, 
        min_length = min_length)
    
    if isinstance(input_data, str):
        text = input_data
    else:
        text = file_preprocessing(input_data)
        
    result = pipe_sum(text)
    result = result[0]['summary_text']
    return result
    


@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)


#streamlit code 
st.set_page_config(layout="wide")

def toggle_button(button_id, new_label):
    return f"""
    <script>
        var button = document.getElementById("{button_id}");
        button.innerHTML = "{new_label}";
    </script>
    """


def display_team_members_info(st):

    col1, col2 = st.columns(2)
    for i, (name, info) in enumerate(members.items(), start=1):
        if i % 2 != 0:
            with col1:
                st.write("")
                st.markdown(f"## {name}")  # Make the name bigger
                st.write("Role:", info["role"])
                st.write(info["description"])
                st.image(info["image_path"], width=250)
                st.write("")
        else:
            with col2:
                st.write("")
                st.markdown(f"## {name}")  # Make the name bigger
                st.write("Role:", info["role"])
                st.write(info["description"])
                st.image(info["image_path"], width=250)
                st.write("")



def get_and_print_summary(st, input_data):
    
    summary = llm_pipeline(input_data)
    st.info("Summarization Complete")
    st.success(summary)
    
    like_button_id = "like_button"
    st.write("Feedback ?")
    if st.button("Like 👍", key=like_button_id, help="Click to like"):
        st.markdown(toggle_button(like_button_id, "Liked 👍"), unsafe_allow_html=True)

    dislike_button_id = "dislike_button"
    if st.button("Dislike 👎", key=dislike_button_id, help="Click to dislike"):
        st.markdown(toggle_button(dislike_button_id, "Disliked 👎"), unsafe_allow_html=True)
    
    

def main():
    
    st.title("Automated Text Summarization")
    st.image(r"C:\Users\saurabhsingh\Downloads\1cover1.jpg", width=1000)

    st.sidebar.header("Menu")
    nav = st.sidebar.radio("Navigation",["HOME","SUMMARIZATION"])

    if nav == "HOME":
        st.subheader("About:")
        st.markdown(markdown_text)
        st.subheader("Team Members:")
        display_team_members_info(st)

                
    elif nav == "SUMMARIZATION":
        choice = st.sidebar.selectbox("Choose data", ["Upload", "Enter Text"])
        
        if choice == "Upload":     
            uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
            st.write("**Disclaimer:** Since the model is not being fine-tuned at runtime, the data entered by the user and the summary generated will not be stored.")
            
            if uploaded_file is not None:
                if st.button("Summarize"):
                    col1, col2 = st.columns(2)
                    filepath = "data/"+uploaded_file.name
                    with open(filepath, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    with col1:
                        st.info("Uploaded File")
                        pdf_view = displayPDF(filepath)   
                    with col2:
                        get_and_print_summary(st, filepath)
 
        else:
            prompt = st.text_area("Enter your text here", height=200)
            st.write("**Disclaimer:** Since the model is not being fine-tuned at runtime, the data entered by the user and the summary generated will not be stored.")
            # st.write(prompt)
            if prompt is not None:
                if st.button("Summarize"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(prompt)
                    with col2:
                        get_and_print_summary(st, prompt)

if __name__ == "__main__":
    main()