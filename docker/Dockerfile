FROM python:3.7

WORKDIR /projects

RUN apt-get update
RUN apt-get install sudo
RUN apt-get install vim -y

RUN mkdir /projects/tads
RUN cd /projects/tads
COPY requirements.txt /projects/tads/
RUN pip install -r /projects/tads/requirements.txt

ENV PYTHONPATH=$PATH:/projects/tads
ENV BETTER_EXCEPTIONS=1