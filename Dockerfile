FROM python:3.8-slim
RUN pip install sentence-transformers
RUN pip install flask
RUN pip install pandas
RUN python -c "from sentence_transformers import SentenceTransformer;"
COPY app.py /app.py
EXPOSE 5000
CMD ["python", "/app.py"]
