
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from diffusers import DiffusionPipeline

# Computer Vision Models

def get_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def classify_image(image_path):
    
    image = Image.open(image_path).convert('RGB')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    inputs = processor(image, return_tensors ='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\\n'.format(float(score))
    return detections

#Natural-Language Processing Models

def text_summarize(user_text):
    
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    inputs = tokenizer(user_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def QA(user_text, context):
    
    processor = AutoTokenizer.from_pretrained("deepset/tinyroberta-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2")
    inputs = processor(user_text, context, return_tensors='pt')
    outputs = model(**inputs)

    starting = outputs.start_logits
    ending = outputs.end_logits

    resp_start = torch.argmax(starting)
    resp_end = torch.argmax(ending) + 1

    answer = processor.convert_tokens_to_string(processor.convert_ids_to_tokens(inputs["input_ids"][0][resp_start:resp_end]))
    return answer;

def similarity(sentences):
    

    #Mean Pooling:

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    processor = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoded_input = processor(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    return similarity_matrix

def gen_image(user_text):
   
    pipeline = DiffusionPipeline.from_pretrained("dataautogpt3/OpenDalleV1.1")
    image = pipeline(prompt = user_text).images[0]
    image.save("generated_image.png")
    return "generated_image.png"

#Note, we are using Pipelining here, instead of the original model because it has some CUDA requirements which I was not
#able to work with properly, so in order to use this function/model, it will take almost ~20GB of local space worth of
#model installation and will require 5-10 minutes in order to process the output.


# From Here, we are Editing the front-view of our website

#------


# Set the page config to wide mode
st.set_page_config(layout="wide")

# Set title
st.title('Multi Purpose AI Model')

# Initialize a list to store the conversation
conversation = []

# Set header
st.header('Welcome! How may I assist you today with?')

col1, col2, col3 = st.columns(3)

# Text input
with col1:
    st.header('Computer Vision')
    st.write('Computer Vision is a field of Artificial Intelligence that trains computers to interpret and understand the visual world.')
    st.header('Please upload the image')
    file = st.file_uploader("Upload Here", type=["jpeg", "jpg", "png"])
    if file:
        st.image(file, use_column_width=50)
    with st.expander("See options"):
        option = st.selectbox('Choose an option', ('Options:', 'Generate Image Caption/Describe Image', 'Classify Image', 'Detect objects and their positions in the image'))
        if file:
            if option == 'Generate Image Caption/Describe Image':
                caption = get_image_caption(file)
                conversation.append(f"Bot 🤖: Sure! Here is a caption for the image you provided: {caption}")
            elif option == 'Classify Image':
                classified_image = classify_image(file)
                conversation.append(f"Bot 🤖: Sure! Here is the classification of the image you provided: {classified_image}")
            elif option == 'Detect objects and their positions in the image':
                detection = detect_objects(file)
                conversation.append(f"Bot 🤖: Sure! These are the objects I was able to detect: {detection}")



with col2:
    st.header('Natural Language Processing')
    st.write('Natural Language Processing is a branch of AI that helps computers understand, interpret and manipulate human language.')
    with st.expander("See options"):
        option = st.selectbox('Choose an option', ('Text Summarisation', 'Question & Answer', 'Search for similarity', 'Generate images'))
        if option == 'Text Summarisation':
            st.header('Please enter the text you want me to summarize:')
            user_text = st.text_area('', height = 80)
            if user_text:
                summary = text_summarize(user_text)
                conversation.append(f"Bot 🤖: Here is the summary for you: \n {summary}")

        elif option == "Question & Answer":
            st.header('Please enter your question')
            user_question = st.text_input('')
            st.header('Please enter the context')
            user_context = st.text_area('', height=50)
            if user_question and user_context:
                answer = QA(user_question, user_context)
                conversation.append(f"Bot 🤖 : Here is your answer:\n {answer}")
            else:
                conversation.append(f"Bot 🤖 : Please fill both question and the context fields properly.")

        elif option == "Search for similarity":
            st.header('Your Original sentence')
            original_sentence = st.text_input('Enter first sentence: ')
            st.header('The first sentence you want to check with')
            first_sentence = st.text_input('Enter second sentence: ')
            st.header('The second sentence you want to check with')
            second_sentence = st.text_input('Enter third sentence: ')
            if original_sentence and first_sentence and second_sentence:
                sentences = [original_sentence, first_sentence, second_sentence]
                similar = similarity(sentences)
                for i in range(len(sentences)):
                    conversation.append(
                        f"Bot 🤖: Here is your sentence:: {sentences[i]}\n Here is your similarity of the sentence with the original sentence: {similar[i][i]:.2f}")
            else:
                conversation.append(f"Bot 🤖: Please fill all the fields. Make sure no field is blank")

        elif option == "Generate images":
            st.header('Give your input')
            user_input = st.text_input('')
            if user_input:
                image_gen = gen_image(user_input)
                conversation.append(f"Bot 🤖: Here is the image generated:")
                st.image(image_gen, use_column_width=100)


# Write agent response


# Display the conversation
st.markdown("## Conversation")
for i in range(len(conversation)):
    if i % 2 == 0:
        st.markdown(f"> {conversation[i]}")
    else:
        st.markdown(f"> 🤖: {conversation[i]}")

# Add a footer
st.markdown("""
---
Made with ♥ by Garv Rajput [IMT2023505]
""")
