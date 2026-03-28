FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_DIR=/workspace
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
ENV PROJ_LIB=/opt/conda/share/proj
ENV GDAL_DATA=/opt/conda/share/gdal
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl pkg-config \
    libgl1-mesa-glx libgtk-3-0 libsm6 libxext6 \
    libstdc++6 && \
    apt-get upgrade -y libstdc++6 && \
    mv /opt/conda/lib/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6.bak || true && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6 && \
    ldconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --set show_channel_urls yes
#Install GDAL
RUN conda install -y -c conda-forge gdal proj && \
    conda clean -afy

#RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
#Install required packages
RUN pip install  --no-cache-dir \
    pandas==2.0.3 \
    opencv-python==4.8.0.76  \
    ultralytics==8.3.133 \
    pycocotools==2.0.7  \
    matplotlib==3.7.2  \
    scikit-learn==1.3.0 \
    numpy==1.26.3

RUN apt-get purge -y build-essential pkg-config && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*;

RUN mkdir -p ${PROJECT_DIR} && chmod -R 777 ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}
CMD ["/bin/bash"]