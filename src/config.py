from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = OUTPUTS_DIR / 'models'
PLOTS_DIR = OUTPUTS_DIR / 'plots'
RESULTS_DIR = OUTPUTS_DIR / 'results'
LOGS_DIR = OUTPUTS_DIR / 'logs'

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 2

DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15

AGE_MIN = 0
AGE_MAX = 116
GENDER_LABELS = {0: 'male', 1: 'female'}


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path
