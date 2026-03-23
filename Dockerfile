FROM python:3.11

WORKDIR /myapp


RUN apt-get update && \
    apt-get install -y curl unzip zstd && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY src/main.py ./main.py

COPY start.sh ./start.sh

RUN pip install -r requirements.txt

# Install Ollama CLI
RUN curl -s https://ollama.com/install.sh | bash

RUN chmod +x ./start.sh

EXPOSE 8000

CMD ["./start.sh"]