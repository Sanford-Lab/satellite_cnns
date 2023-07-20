FROM apache/beam_python3.10_sdk:2.48.0

MAINTAINER dannie.daley@yale.edu


# Prebuilt dependencies
#ADD /benin-data /benin-data
#ADD /benin-model /benin-model
#COPY requirements.txt /tmp/base_requirements.txt

RUN pip install tensorflow
#RUN pip install -r /tmp/base_requirements.txt


# Set the entrypoint to the Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]