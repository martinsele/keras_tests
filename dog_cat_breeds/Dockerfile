FROM python:3.6-alpine

ENV SERVER_PORT=8088
EXPOSE $SERVER_PORT

RUN apk update && \
	apk add --no-cache \
    # bash \
    libstdc++ \
    libgomp \
    build-base \
    cmake \
    gfortran \
    libpng && \
    ln -s locale.h /usr/include/xlocale.h && \
    apk add --no-cache --virtual .build-deps \
    lapack-dev \
    musl-dev \
    python3-dev \
    jpeg-dev \
    freetype-dev \
    libffi-dev \
    openssl-dev \
    g++ && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    tensorflow==1.15 \
    Keras==2.2.4 \
    opencv-python \
    Flask \
    wheel \
    lightgbm

# RUN apk del .build-deps

COPY . .

ENTRYPOINT [ "python", "cloud.py" ]