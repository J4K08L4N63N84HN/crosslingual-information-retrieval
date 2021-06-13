Crosslingual information retrieval
-----------------------------------------

.. image:: https://img.shields.io/github/stars/J4K08L4N63N84HN/crosslingual-information-retrieval?style=social   :alt: GitHub Repo stars
.. image:: https://img.shields.io/github/repo-size/J4K08L4N63N84HN/crosslingual-information-retrieval?style=social   :alt: GitHub repo size
.. image:: https://img.shields.io/github/stars/J4K08L4N63N84HN/crosslingual-information-retrieval?style=social   :alt: GitHub Repo stars


Cross-Lingual Information Retrieval is the task of getting information in a different language than the original query. Our goal is to implement a lightweight system, unsupervised and supervised, to recognize the translation of a sentence in a large collection of documents in a different language. Testing different cross-lingual word embedding- and text-based features with wide-ranging parameter combinations, our best model, the MLPClassifier, achieved a Mean Average Precision of 0.8459 on our English-German test collection. Our lightweight system also demonstrates zero-shot performance in other languages, such as Italian and Polish. We compare our results to the SOTA, but resource-hungry transformer model XLM-R.


Table of Contents
#################

.. contents::

Description
#################



How to Install
##############

To use this code you have to follow these steps:

1. Start by cloning this Git repository:

.. code-block::

    $  git clone https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval.git
    $  cd crosslingual-information-retrieval

2. Continue by creating a new conda environment (Python 3.8):

.. code-block::

    $  conda env create --file environment.yaml
    $  conda activate crosslingual-information-retrieval

3. Set the conda environment on your jupyter notebook:

.. code-block::

    $ python -m ipykernel install --user --name=crosslingual-information-retrieval


How to Use
##########

The repository serves two functions: Inducing a cross-lingual word embedding and using different models for the translation retrieval task.

Detailed documentation and usage instructions can be found `here <https://crosslingual-information-retrieval.readthedocs.io/en/latest/>`__.


Models
######

We make all our trained models available in this `Drive <https://drive.google.com/drive/folders/1r0UExZMI46dbYx_zfdVCmbPNJC3O8yU9?usp=sharing/>`__.

Furthermore, you can download all related data in the following `Drive <https://drive.google.com/drive/folders/1EuDDZSmv2DWgw3itdGSDwKz3UYIcLVmT?usp=sharing/>`__. 

Credits
#######

The project started in March 2021 as a Information Retrieval project at the University of Mannheim. The project team consists of:

* `Minh Duc Bui <https://github.com/MinhDucBui/>`__
* `Jakob Langenbahn <https://github.com/J4K08L4N63N84HN/>`__
* `Niklas Sabel <https://github.com/NiklasSabel/>`__

Reference
#########

License
#######

This repository is licenced under the MIT License. If you have any enquiries concerning the use of our code, do not hesitate to contact us.







