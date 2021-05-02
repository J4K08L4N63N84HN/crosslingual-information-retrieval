# crosslingual-information-retrieval

## Possible Features

### Sentence based features
- number of words as feature
- differences of words absolute and relative with respect to smaller number as base
- total number of chars
- number of unique words
- number of stopwords
- sentiment analysis

### Grammar based features
- pos-tagger and get number of nouns, verbs, adjectives
- verb times

### Word based features
- number of characters in words 
- average char per word
- absolute difference between avg char
- amount of direct translations
- average numbers of letters per word
- word density (word count / char count)
- named entities (Jakob)
- named numbers

### Word embedding based features
- word embeddings (average, weighted average etc.)

### Punctuation features
- number of punctuation marks / differences of punctuation marks absolute and relative with respect to smaller number as base
- number of question marks
- number of exclamation marks
- number of commas
- number of semicolons
- number of colons
- number of ellipsis
- number of apostrophes
- number of hyphens 
- number of quotation marks
- number of slashes
- number of brackets
- number of special characters



## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
