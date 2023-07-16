from setuptools import setup

# Requirements for the Dataflow dataset creation pipeline.
setup(
    name="SPIRES-classifcation",
    url="https://github.com/Sanford-Lab/satellite_cnns/blob/main/benin_apache_pipeline.ipynb",
    packages=["src/benin-data"],
    install_requires=[
        "apache-beam[gcp]==2.46.0",
        "earthengine-api==0.1.358",
        "tensorflow==2.12.0",
    ],
)