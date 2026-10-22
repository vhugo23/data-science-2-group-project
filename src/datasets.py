import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIRS = {
    "asvspoof19_la": os.getenv("ASVSPOOF19_LA_DIR"),
    "asvspoof21_la": os.getenv("ASVSPOOF21_LA_DIR"),
    "in_the_wild": os.getenv("IN_THE_WILD_DIR"),
    "scam_dialogue": os.getenv("SCAM_DIALOGUE_DIR"),
}


def dataset_path(name):
    path = DATA_DIRS[name]
    return Path(path).expanduser() if path else None


if __name__ == "__main__":
    for name, value in DATA_DIRS.items():
        path = dataset_path(name)
        print(f"{name}: {path if path and path.exists() else 'not configured or missing'}")
