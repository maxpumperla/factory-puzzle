FROM python:3.7

#FROM tensorflow/tensorflow:nightly-gpu-py3
#
#RUN apt-get update \
#    && apt-get install -y \
#        curl \
#        tmux \
#        screen \
#        rsync \
#        apt-transport-https \
#        zlib1g-dev \
#        libgl1-mesa-dev \
#        git \
#        wget \
#        cmake \
#        build-essential \
#        curl \
#        unzip \
#    && apt-get clean \
#    && echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
#    && wget \
#        --quiet 'https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh' \
#        -O /tmp/anaconda.sh \
#    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
#    && rm /tmp/anaconda.sh \
#    && /opt/conda/bin/conda install -y \
#        libgcc \
#    && /opt/conda/bin/conda clean -y --all \
#    && /opt/conda/bin/pip install \
#        flatbuffers \
#        cython==0.29.0 \
#        numpy==1.15.4
#ENV PATH "/opt/conda/bin:$PATH"
#RUN conda remove -y --force wrapt
#RUN pip install -U pip
## AttributeError: 'numpy.ufunc' object has no attribute '__module__'
#RUN /opt/conda/bin/pip uninstall -y dask
#ENV PATH "/opt/conda/bin:$PATH"
## For Click
#ENV LC_ALL=C.UTF-8
#ENV LANG=C.UTF-8
#RUN pip install gym[atari]==0.10.11 opencv-python-headless lz4 pytest-timeout smart_open torch torchvision
#RUN pip install --upgrade bayesian-optimization
#RUN pip install --upgrade hyperopt==0.1.2
#RUN pip install ConfigSpace==0.4.10
#RUN pip install --upgrade sigopt nevergrad scikit-optimize hpbandster lightgbm xgboost tensorboardX
#RUN pip install -U mlflow
#RUN pip install -U pytest-remotedata>=0.3.1
#
#
## install node and build dashboard
#RUN curl -sL https://deb.nodesource.com/setup_13.x | bash -
#RUN apt-get install -y nodejs
#RUN git clone --single-branch --branch master https://github.com/ray-project/ray.git
#WORKDIR /ray/python
#RUN cd ray/dashboard/client && npm ci && npm run build


WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8080
COPY . /app
RUN cd /app && python setup.py install && cd -
CMD streamlit run --server.port 8080 --server.enableCORS false serving_app.py
