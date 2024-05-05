# Multi-Purpose AI Model with Computer Vision and NLP

This project is an interactive AI application capable of performing various tasks related to both Computer Vision and Natural Language Processing (NLP). It incorporates state-of-the-art machine learning models to provide functionalities such as image captioning, object detection, text summarization, question answering, semantic similarity measurement, and image generation from textual prompts.

## Features

### Computer Vision:
- **Image Captioning:** Generates descriptive captions for uploaded images.
- **Image Classification:** Classifies images into predefined categories.
- **Object Detection:** Detects objects within images and provides their positions.

### Natural Language Processing:
- **Text Summarization:** Summarizes long text passages into concise summaries.
- **Question & Answering:** Answers questions based on provided contexts.
- **Semantic Similarity:** Measures the similarity between provided sentences.
- **Image Generation from Text:** Generates images based on textual prompts.

## Technology Used

- **Machine Learning Frameworks:** PyTorch
- **Web Application Framework:** Streamlit
- **Image Processing Library:** PIL (Python Imaging Library)
- **CUDA (Optional):** For GPU acceleration (if available)

## Libraries Used

- **Transformers:** Used for accessing pre-trained state-of-the-art NLP models.
- **PIL (Python Imaging Library):** Used for image manipulation and processing.
- **torch:** PyTorch library for deep learning.
- **Streamlit:** Framework for building interactive web applications.
- **scikit-learn:** Used for cosine similarity calculation.
- **diffusers:** Library for generating images based on textual prompts.

## Installation

### Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run main.py
    ```

## Note

- The AI models utilized in this project require significant computational resources and may take some time to process requests, especially for tasks like image generation.
- Ensure that the required CUDA dependencies are installed if GPU acceleration is available.

## Credits

- **Author:** Garv Rajput ([IMT2023505](https://github.com/imt2023505))
- **Libraries:** Transformers, PIL, torch, Streamlit, scikit-learn, diffusers
- **Models:** BLIP, DETR, ViT, Pegasus, TinyRoberta, Sentence Transformers, OpenDalle
