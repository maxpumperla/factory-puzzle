FROM python:3.7

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8080
COPY . /app
RUN cd /app && python setup.py install && cd -
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
