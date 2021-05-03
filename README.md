# crosslingual-information-retrieval

## Possible Features

# write frist feature generation function
    Niklas

### Sentence based features
- number of words as feature (Niklas) -> done
- differences of words absolute and relative with respect to smaller number as base (Niklas) -> done
- total number of chars (Niklas) -> done
- number of unique words (Niklas) -> done
- number of stopwords (Niklas) -> done
- sentiment analysis (Jakob)

### Grammar based features
- pos-tagger and get number of nouns, verbs, adjectives (Niklas) -> done
- verb times (Niklas)

### Word based features
- number of characters in words (Niklas) -> done
- average char per word (Niklas) -> done
- absolute difference between avg char (Niklas) -> done
- named numbers (Niklas) -> done (included in POS Tagger NUM)
- amount of direct translations (Jakob)
- named entities (Jakob)


### Word embedding based features
- word embeddings (average, weighted average etc.) (Jakob)

### Punctuation features
- number of punctuation marks (Niklas) delete real sentence ending points, cause sometimes two sentences in one language will be mapped to one sentence in the other language -> done
- number of question marks (Niklas)-> done
- number of exclamation marks (Niklas)-> done
- number of commas (Niklas)-> done
- number of semicolons (Niklas)-> done
- number of colons (Niklas) -> done
- number of ellipsis (Niklas) -> done
- number of apostrophes (Niklas) -> done
- number of hyphens (Niklas) -> done
- number of quotation marks (Niklas)-> done
- number of slashes (Niklas) -> done
- number of brackets (Niklas) -> done
- number of special characters (Niklas) -> done



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
