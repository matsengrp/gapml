FROM tensorflow/tensorflow:1.9.0-py3

RUN pip install --upgrade pip
RUN pip install biopython==1.70
RUN pip install botocore==0.29.0
RUN pip install ete3==3.0.0b35
RUN pip install matplotlib==2.1.0
RUN pip install numpy==1.13.3
RUN pip install pandas==0.20.3
RUN pip install scipy==0.19.1
RUN pip install seaborn==0.8.1
RUN pip install six==1.11.0

RUN apt-get update && apt-get install -y r-base
RUN R -e "install.packages('ape', repos = 'http://cran.us.r-project.org')"

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:webupd8team/java -y
RUN apt-get update
RUN echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install -y oracle-java8-installer
RUN apt-get clean

COPY run_script.sh /usr/local/bin/run_script.sh
WORKDIR "/tmp"
ENTRYPOINT ["/usr/local/bin/run_script.sh"]
