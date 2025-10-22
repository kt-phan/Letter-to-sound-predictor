# Only run if the dataset is missing

import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('rtatman/cmu-pronouncing-dictionary', path='.', unzip=True)

