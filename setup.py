from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Crosslingual Information Retrieval',
    author='Duc, Jakob, Niklas',
    license='',
    install_requires=[
        "pandas==1.2.4",
        "click~=7.1.2",
        "nltk~=3.6.1",
        "setuptools~=56.2.0",
        "numpy~=1.20.2",
        "spacy~=3.0.6",
        "jupyter~=1.0.0",
        "matplotlib~=3.4.2",
        "seaborn~=0.11.1",
        "scikit-learn~=0.24.2",
        "wmd~=1.3.2",
        "scipy~=1.6.2",
        "requests~=2.25.1",
        "sphinx~=4.0.2",
        "tqdm~=4.60.0",
        "docutils~=0.16",
        "sklearn~=0.24.2",
        "xgboost~=1.4.2",
        "transformers~=4.6.1",
        "torch~=1.8.1",
        "datasets~=1.8.0",
        "pickle5~=0.0.11"
            ]
)
