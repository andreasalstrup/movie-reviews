FROM ambolt/emily:1.0.0-torch-nlp

COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace

# add installation of Debian packages here

# Remote development
RUN apt-get update --allow-releaseinfo-change && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Password & Authentication

# Permit login as root
RUN sed -i 's/PermitRootLogin prohibit-password/#PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

# Don't require password
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Declare the authorized keys as the file we're copying in ourselves
RUN sed -i 's/^AuthorizedKeysFile.*$/AuthorizedKeysFile \/pycharm\/.ssh\/authorized_keys/' /etc/ssh/sshd_config
RUN echo "RSAAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# SSH & Keeping Session Alive 
RUN sed 's@sessions*requireds*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# add user with normal permission
RUN adduser --disabled-password --gecos "" --home /pycharm --force-badname movie-reviews
COPY .ssh/authorized_keys /pycharm/.ssh/authorized_keys

RUN chmod 755 /pycharm/.ssh
RUN chmod 644 /pycharm/.ssh/authorized_keys

# add user to sudo group
RUN adduser movie-reviews sudo

# set no password to sudo group
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make SSH script executable
ADD .ssh/launch_ssh.sh /
RUN chmod +x /launch_ssh.sh

# set emilyuser as current user
USER movie-reviews

RUN pip install --disable-pip-version-check -r requirements.txt

# Run SSH
CMD ["/launch_ssh.sh"]