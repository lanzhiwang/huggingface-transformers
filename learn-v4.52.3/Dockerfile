FROM python:3.12.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            git tree net-tools openssh-server vim \
            ca-certificates \
            gnupg \
            build-essential gdb && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /run/sshd && \
    chmod 0755 /run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN echo "root:password" | chpasswd

EXPOSE 22

ENTRYPOINT /usr/sbin/sshd -D
