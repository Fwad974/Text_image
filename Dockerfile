FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
MAINTAINER Fwad Abdi <mfabdi014@gmail.com>

RUN apt update && apt install cmake make git gcc -y
WORKDIR /app

COPY ./ /app

RUN pip install --requirement ./req.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80","--workers" ,"8" ]
