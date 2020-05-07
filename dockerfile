FROM jupyter/scipy-notebook

RUN pip install \
    'tensorflow-gpu==1.15.0' && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN pip install nltk
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('vader_lexicon')"

RUN pip install scikit-learn bert-tensorflow tensorflow-hub

# RUN python -c "from cayde.well.nlpwell import create_tokenizer_from_hub_module; create_tokenizer_from_hub_module()"

WORKDIR "/home/project"

# ENTRYPOINT [ "bash" ]