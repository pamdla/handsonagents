FROM python:3.12-slim
LABEL maintainer="agent1237@datawhalechina.cn"
LABEL team="agent①②③⑦"
LABEL project="handsonagents"
LABEL git.repo="https://github.com/pamdla/handsonagents"

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /learning

COPY requirements.txt .

RUN rm -f /etc/apt/sources.list.d/* && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main" >> /etc/apt/sources.list

# RUN apt update \
#     && apt install -y --no-install-recommends \
#     && apt install -y git wget make gcc g++ \
#     && apt autoremove -y \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt
