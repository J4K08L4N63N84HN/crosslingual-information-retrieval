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

We make all our code availabe that were used for this project. It contains the data preprocessing, inducing cross-lingual word embeddings, training and evaluating all models. You can find the code for each part in the following table: 

*  `Data Preprocessing <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/data/>`__
*  `Feature Generation <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/features>`__
*  `Inducing CLWE <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/embeddings>`__
*  `Training and Evaluating Supervised Models <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/models>`__

All Experiments done were written in Jupyter Notebooks, which can be found in this `Folder <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/notebooks>`__

Furthermore, we make all models available `Drive <https://drive.google.com/drive/folders/1r0UExZMI46dbYx_zfdVCmbPNJC3O8yU9?usp=sharing/>`__. All raw and preprocessed data can be downloaded in the following `Drive <https://drive.google.com/drive/folders/1EuDDZSmv2DWgw3itdGSDwKz3UYIcLVmT?usp=sharing/>`__. 

Our results are summarized in the following table:


.. image:: https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/blob/main/reports/figures/final_results.png

How to Install
##############

To use this code you have to follow these steps:

1. Start by cloning this Git repository:

.. code-block::

    $  git clone https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval.git
    $  cd crosslingual-information-retrieval

2. Continue by creating a new conda environment (Python 3.8):

.. code-block::

    $  conda create -n animate_logos python=3.8
    $  conda activate animate_logos

3. Install the dependencies:

.. code-block::

    $ pip install -r requirements.txt

For a detailed documentation you can refere to `here <https://crosslingual-information-retrieval.readthedocs.io/en/latest/index.html>`__ or create your own sphinx documentation with

Credits
#######

The project started in March 2021 as a Information Retrieval project at the University of Mannheim. The project team consists of:

* `Minh Duc Bui <https://github.com/MinhDucBui/>`__
* `Jakob Langenbahn <https://github.com/J4K08L4N63N84HN/>`__
* `Niklas Sabel <https://github.com/NiklasSabel/>`__

License
#######

This repository is licenced under the MIT License. If you have any enquiries concerning the use of our code, do not hesitate to contact us.







