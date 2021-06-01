.. crosslingual-information-retrieval documentation master file
$ sphinx-apidoc -F -o doc/ src/my_project/
$ cd doc
$ make html
.. include:: ../README.rst

.. toctree::
   :maxdepth: 2
   :caption: Content

   source/src.data
   source/src.features
   source/src.models
   source/src.utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
