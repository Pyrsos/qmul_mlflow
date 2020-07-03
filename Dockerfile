FROM continuumio/miniconda3:4.8.2

RUN pip install mlflow>=1.9 \
    && pip install azure-storage-blob \
    && pip install numpy \
    && pip install scipy \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install cloudpickle
