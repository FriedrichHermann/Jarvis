
FROM jupyter/scipy-notebook

# install build utilities 
RUN pip install joblib

# check our python environment
RUN python3 –version
RUN pip3 –version

# set the working directory for containers
WORKDIR /usr/src/my-app-name

# Copy all the necessary files to the image (COPY <src> … <dest>)
COPY train.py ./train.py
COPY inference.py ./inference.py

# Run python scripts 
RUN python3 train.py
RUN python3 inference.py