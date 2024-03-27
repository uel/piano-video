FROM python:3.10

WORKDIR /PianoVideo

COPY requirements.txt /PianoVideo/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y ffmpeg

COPY src /PianoVideo/src

CMD ["python", "src/processing.py"]
