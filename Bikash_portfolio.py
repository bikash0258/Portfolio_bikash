import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie  
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# 🔁 Load Lottie animation function
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ✅ Load coder animation
lottie_coder = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_UBiAADPga8.json")

st.write("##")
st.subheader("Hello and welcome!")
st.title("Bikash Sharma | Aspiring Data Scientist & Software Developer")

with st.container():
    selected = option_menu(
        menu_title=None,
        options=['About', 'Projects', 'Skills', 'Contact'],
        icons=['person', 'code-slash', 'chat-left-text-fill'],
        orientation='horizontal'
    )

# ================= About Section =================
if selected == 'About':
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("##")
            st.subheader("Final-year M.Tech Student | NIT Silchar")
            st.write("""
               Passionate about Computer Vision, Deep Learning, and Intelligent Systems.
               Hands-on experience in building real-world ML applications including
               medical imaging, object detection, and surveillance-based solutions.
               """)
        with col2:
            st_lottie(lottie_coder, height=200, key="coder")

        st.write("___")

        with st.container():
            col3, col4 = st.columns(2)
            with col3:
                st.header("🎓 Education")

                st.subheader("National Institute of Technology, Silchar")
                st.write("**Degree:** M.Tech in Computer Science & Engineering (AI)")
                st.write("**Grade:** 7.79 CGPA")
                st.write("**Duration:** 2024 – 2026")

                st.markdown("---")

                st.subheader("Assam Science & Technology University")
                st.write("**College:** Girijananda Chowdhury Institute of Management and Technology")
                st.write("**Degree:** B.Tech in Computer Science & Engineering")
                st.write("**Grade:** 8.41 CGPA")
                st.write("**Duration:** 2020 – 2024")

            with col4:
                st.header("Experience")

                st.subheader("🎓 Android Real-Time Face Detection App — IIT Guwahati")
                st.write("**Duration:** July 2023 – Sept 2023")
                st.markdown("""
                - Developed an advanced Android application using **Java** (backend) and **XML** (UI).
                - Integrated **Firebase ML Kit** for real-time face detection and recognition.
                - Achieved smooth and responsive performance on Android devices.
                """)

                st.markdown("---")

                st.subheader("💳 Credit Card Risk Score Prediction — Internshala")
                st.write("**Date:** Sept 9, 2022")
                st.markdown("""
                - Built a **credit risk model** to predict default probability for IDFC Bank customers.
                - Applied **feature engineering**, handled **class imbalance**, and performed **dimensionality reduction**.
                - Tools used: Python, NumPy, scikit-learn, PyTorch, Pandas, Matplotlib.
                """)

# ================= Projects Section =================
elif selected == 'Projects':
    with st.container():
        st.header("🚀 Projects")
        st.write("")

        # Project 1 - BERT
        st.subheader("🧠 NLP-Based Mental Health Predictor (BERT)")
        st.markdown("""
        - Developed a web app to classify user input text into mental health conditions using a fine-tuned **BERT** model.
        - Designed an end-to-end pipeline including **text preprocessing**, **model training**, and **deployment** in Streamlit.
        - Leveraged **PyTorch**, **Hugging Face Transformers**, **NLTK**, and **scikit-learn** for efficient training and evaluation.
        - Delivered accurate predictions and an interactive interface for real-world usability.
        - Tools: PyTorch, Hugging Face, NLTK, Streamlit, scikit-learn.
      #  - 📂 [View GitHub Repository](https://github.com/bikash0258/Mental_Health_BERT)
        """)

        st.markdown("---")

        # Project 2 - T5
        st.subheader("🍳 Custom Recipe Generator using T5 Transformer")
        st.markdown("""
        - Developed a deep learning application to generate **personalized cooking recipes** using the **T5 Transformer**.
        - Implemented NLP pipeline: preprocessing, tokenization, and sequence-to-sequence generation with **Hugging Face Transformers** and **PyTorch**.
        - Fine-tuned **T5-base model** on a custom dataset of ingredients and instructions to translate inputs into structured recipe steps.
        - Optimized for **BLEU score** and inference quality to ensure recipe accuracy.
        - Tools: PyTorch, Hugging Face, Transformers, NLP, Streamlit.
     #   - 📂 [View GitHub Repository](https://github.com/bikash0258/Recipe_Generator_T5)
        """)

        st.markdown("---")

        # Project 3 - Breast Cancer
        st.subheader("🧬 Breast Cancer Diagnosis Using Hybrid Deep Learning")
        st.markdown("""
        - Developed a high-performance diagnostic model using hybrid deep learning techniques.
        - Compared multiple architectures: `ResNet50 + SVM`, `ResNet50 + InceptionV3`, `ResNet50 + MobileNet`, and `ResNet50 + Perceptron`.
        - Final model (ResNet + Perceptron) achieved **97.5% accuracy** on MRI-based breast cancer classification.
        - Tools: Python, Keras, TensorFlow, Albumentations, Matplotlib.
        - 📂 [View GitHub Repository](https://github.com/bikash0258/---Breast-Cancer-Diagnosis-Using-Hybrid-Deep-Learning-Models)
        """)

        st.markdown("---")

        # Project 4 - Vehicle
        st.subheader("🚗 Vehicle Image Classification")
        st.markdown("""
        - Built a multi-class vehicle classifier using a custom image dataset.
        - Applied data cleaning with **CleanVision**, augmentation using **Albumentations**, and class balancing with **WeightedRandomSampler**.
        - Utilized **EfficientNet-B0** with transfer learning for classification.
        - Achieved high validation accuracy and robust performance across diverse vehicle classes.
        - Tools: PyTorch, torchvision, matplotlib, pandas.
        - 📂 [View GitHub Repository](https://github.com/bikash0258/Vehicle_Classification/tree/main)
        """)

        st.markdown("---")

        # Project 5 - Disaster
        st.subheader("🌊 Unsupervised Segmentation for Disaster Management")
        st.markdown("""
        - Designed an unsupervised image segmentation model to detect **flood-affected regions**.
        - Applied clustering algorithms: **K-Means**, **GMM**, and **DBSCAN** on 290 annotated flood images.
        - Used **Elbow Method** and **Silhouette Score** for evaluation.
        - Helped automate mapping of disaster zones for emergency response systems.
        - Tools: Python, OpenCV, scikit-learn, matplotlib.
        - 📂 [View GitHub Repository](https://github.com/bikash0258/unsupervised-flood-segmentation)
        """)

# ================= Skills Section =================
elif selected == 'Skills':
    with st.container():
        st.header("🛠️ Skills Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💻 Languages & Frameworks")
            st.markdown("""
            - **Programming:** C, C++, Python  
            - **Libraries & Tools:** Pandas, NumPy, Matplotlib, Seaborn, Plotly, hvPlot  
            - **ML/DL Frameworks:** TensorFlow, Keras, PyTorch, OpenCV  
            - **Web & UI:** Streamlit  
            - **Database:** MySQL
            """)

            st.subheader("🧠 Core Competencies")
            st.markdown("""
            - Data Structures & Algorithms (DSA)  
            - Operating Systems  
            - DBMS  
            - Deep Learning  
            - Artificial Intelligence (AI)  
            - Machine Learning (ML)  
            - Statistics  
            - Image Processing
            """)

        with col2:
            st.subheader("📦 Software & IDEs")
            st.markdown("""
            - C/C++ Compilers  
            - Android Studio  
            - Visual Studio  
            - IntelliJ IDEA  
            - PyCharm
            """)

            st.subheader("🔬 Additional Knowledge")
            st.markdown("""
            - Computer Vision  
            - Automatic Number Plate Recognition (ANPR)  
            - Smart City Technologies  
            - Object Detection (YOLO, SSD)
            """)

            st.subheader("🤝 Soft Skills")
            st.markdown("""
            - Problem Solving  
            - Adaptability  
            - Teamwork  
            - Active Listening  
            - Quick Learner  
            - Self-Taught & Team-Oriented
            """)

# ================= Contact Section =================
elif selected == 'Contact':
    with st.container():
        st.header("📬 Get in Touch")
        st.write("Fill out the form below to send me a message directly:")

        contact_form = """
        <form action="https://formsubmit.co/bikashsharma12dec@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">

            <input type="text" name="name" placeholder="Your Name" required 
                style="width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc;">

            <input type="email" name="email" placeholder="Your Email" required 
                style="width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc;">

            <textarea name="message" placeholder="Your Message..." required 
                style="width: 100%; padding: 10px; height: 150px; border-radius: 5px; border: 1px solid #ccc;"></textarea>

            <button type="submit" 
                style="margin-top: 10px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;">
                Send
            </button>
        </form>
        """

        components.html(contact_form, height=600)

