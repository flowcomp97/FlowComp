from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, CONFIGS_DIR, OUTPUTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def output_path(*parts: str) -> Path:
    ensure_project_dirs()
    return OUTPUTS_DIR.joinpath(*parts)


def log_path(*parts: str) -> Path:
    ensure_project_dirs()
    return LOGS_DIR.joinpath(*parts)
