FROM continuumio/miniconda3

# Install python packages
RUN mkdir /opt/api
COPY requirements.txt /opt/api/
RUN pip install -r /opt/api/requirements.txt

# Copy files into container
COPY model /opt/api/model
COPY h_d_demo.py /opt/api/
COPY *.png /opt/api/
COPY *.jpg /opt/api/

# Set work directory and open the required port
WORKDIR /opt/api
EXPOSE 8501

# Run our service script
CMD ["streamlit", "run","h_d_demo.py"]
