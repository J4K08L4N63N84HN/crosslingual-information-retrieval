# crosslingual-information-retrieval

## Possible Features

### Sentence based features
- number of words as feature (Niklas)
- differences of words absolute and relative with respect to smaller number as base (Niklas)
- total number of chars (Niklas)
- number of unique words (Niklas)
- number of stopwords (Niklas)
- sentiment analysis (Jakob)

### Grammar based features
- pos-tagger and get number of nouns, verbs, adjectives (Niklas)
- verb times (Niklas)

### Word based features
- number of characters in words (Niklas)
- average char per word (Niklas)
- absolute difference between avg char (Niklas)
- named numbers (Niklas)
- amount of direct translations (Jakob)
- named entities (Jakob)


### Word embedding based features
- word embeddings (average, weighted average etc.) (Jakob)

### Punctuation features
- number of punctuation marks (Niklas) delete real sentence ending points
- number of question marks (Niklas)
- number of exclamation marks (Niklas)
- number of commas (Niklas)
- number of semicolons (Niklas)
- number of colons (Niklas)
- number of ellipsis (Niklas)
- number of apostrophes (Niklas)
- number of hyphens (Niklas)
- number of quotation marks (Niklas)
- number of slashes (Niklas)
- number of brackets (Niklas)
- number of special characters (Niklas)



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
