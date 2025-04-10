from setuptools import setup, find_packages

setup(
    name="Summarization Project",
    version="0.1",
    packages=find_packages(exclude=["rouge"]),
    install_requires=[], 
)
