# Text track

**Owner:** Justin Kondratenko

Keep text cleaning, TF-IDF and sentence-embedding baselines, scam-intent classifiers, and cross-dataset text evaluation here. Put track notebooks under `notebooks/text/`.

Export one row per dialogue with a stable dialogue ID, shared split name (`train`, `validation`, or `test`), scam label, and model score. Preserve the dataset's 1,280/320 train/test split; draw validation data from its training split and exclude the scam-type column from model inputs.
