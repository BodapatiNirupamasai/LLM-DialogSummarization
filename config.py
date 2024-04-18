
model_name='google/flan-t5-base'
huggingface_dataset_name = "knkarthick/dialogsum"
output_dir = './peft-dialogue-summary-training'
peft_model_path="./peft-model-final"

loaded_model = 'nirupama1899/dialog-summarization'


members = {
    "Saurabh Singh": {
        "role": "ML Engineer/UI Developer",
        "image_path": r"C:\Users\saurabhsingh\Downloads\Saurabh singh t.png",
        "description": "Saurabh is responsible for developing and optimizing machine learning models for text summarization. He also designs and implements the user interface of the application."
    },
    "Nirupama Bodapati": {
        "role": "ML Engineer",
        "image_path": r"C:\Users\saurabhsingh\Downloads\nirupama_bodappati.png",
        "description": "Nirumpama focuses on the machine learning aspects of the application, working on data preprocessing, model training, and evaluation."
    },
    "Sanath Thumu": {
        "role": "Business Analyst",
        "image_path": r"C:\Users\saurabhsingh\Downloads\sanath thumu.png",
        "description": "Sanath analyzes business requirements and ensures that the application aligns with stakeholders' needs and business goals."
    },
    "Prashanth Kumar Gunda": {
        "role": "Data Scientist",
        "image_path": r"C:\Users\saurabhsingh\Downloads\prashant kumar.png",
        "description": "Prashanth handles data processing for the application, gathering and preprocessing datasets used to train machine learning models."
    }
}



markdown_text = """
        The **Automated Text Summarization** application specializes in simplifying dialogue transcript summarization, offering users comprehensive capabilities for summarizing text documents. Users can easily input their text manually or upload PDF documents, enabling a wide range of use cases and enhancing productivity. 

        ### Key Features:
        - **Flexible Input Options:** Users can input text manually or upload PDF documents for summarization, providing versatility and convenience.
        - **Advanced Machine Learning Models:** The application harnesses cutting-edge machine learning models (LLMs) to generate accurate and concise summaries tailored to the input text.
        - **Tailored Summaries:** Summaries produced by the application are customized to the specific content provided by the user, ensuring relevance and precision.
        - **Enhanced Productivity:** By automating the summarization process, the application streamlines tasks related to information extraction, saving users valuable time and effort.
        - **User-Friendly Interface:** With an intuitive user interface, the application offers a seamless and efficient user experience, allowing users to quickly access and utilize its summarization capabilities.

        **Automated Text Summarization** empowers users to efficiently summarize dialogue transcripts and text documents, facilitating better understanding and decision-making.
        """