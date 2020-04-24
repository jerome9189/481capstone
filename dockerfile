FROM jupyter/tensorflow-notebook

RUN pip install nltk
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

RUN pip install scikit-learn

# ENTRYPOINT [ "bash" ]