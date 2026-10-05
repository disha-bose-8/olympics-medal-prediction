'''from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "olympics_merged.csv"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [1988, 1992, 1996, 2000, 2004, 2008]
VALIDATION_YEAR = 2012
TEST_YEAR = 2016
TARGET = "Medals"

FEATURES = [
    "Year",
    "GDP_Growth",
    "GDP_Per_Capita",
    "Pop",
    "Pop_pct_World",
    "Athletes",
    "Athletes_pct",
    "Medals_Last_Games",
    "Total_Medals_Year",
]
'''

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "olympics_merged.csv"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [1988, 1992, 1996, 2000, 2004, 2008]
VALIDATION_YEAR = 2012
TEST_YEAR = 2016
TARGET = "Medals"

# The paper's full feature set (Table 1): 13 features.
FEATURES = [
    "Year",
    "GDP",
    "GDP_Per_Capita",
    "GDP_Growth",
    "GDP_pct_World",
    "Pop",
    "Pop_pct_World",
    "Pop_Growth",
    "Area",
    "Athletes",
    "Athletes_pct",
    "Medals_Last_Games",
    "Total_Medals_Year",
]