# Use the Triton server image as the base
FROM nvcr.io/nvidia/tritonserver:22.08-py3

# Install the required Python libraries
RUN pip install torch==2.0.1 \
    && pip install torch-utils \
    && pip install textstat==0.7.3 \
    && pip install transformers==4.31.0 \
    && pip install einops \
    && pip install accelerate \
    && pip install xformers \
    && pip install numpy==1.24.3 \
    && pip install requests==2.28.1 \
    && pip install PyPDF2==3.0.1 \
    && pip install pyppeteer \
    && pip install spacy \
    && pip install scikit-learn \
    && pip install https://huggingface.co/spacy/en_core_web_lg/resolve/main/en_core_web_lg-any-py3-none-any.whl
