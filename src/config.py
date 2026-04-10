from pathlib import Path
# all parameters and constants used across the project should be defined here

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"
OPENCV_MODELS_DIR = PROJECT_ROOT / "models" / "opencv"

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 2

AGE_MIN = 0
AGE_MAX = 116
GENDER_LABELS = {0: "male", 1: "female"}

# Used for fair comparison with the common OpenCV / Adience age classifier.
AGE_BUCKETS = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100),
]

AGE_BUCKET_LABELS = [f"{a}-{b}" for a, b in AGE_BUCKETS]
# Safer for spreadsheet apps that like converting 0-2 into dates.
AGE_BUCKET_LABELS_EXCEL = [f'="{label}"' for label in AGE_BUCKET_LABELS]
AGE_BUCKET_MIDPOINTS = [(a + b) / 2.0 for a, b in AGE_BUCKETS]
NUM_AGE_BUCKETS = len(AGE_BUCKETS)

AGE_GROUPS = {
    "child": (0, 12),
    "teen": (13, 19),
    "young_adult": (20, 35),
    "adult": (36, 59),
    "senior": (60, 116),
}
