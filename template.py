import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "cnnClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/main.yaml",
    "config/config.yaml",
    "model/",
    "research/01_data_ingestion.ipynb",
    "research/02_prepare_base_model.ipynb",
    "research/03_model_trainer.ipynb",
    "research/04_model_evaluation_with_mlflow.ipynb",
    "research/trials.ipynb"
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/model_evaluation_mlflow.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/prepare_base_model.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/pipeline/stage1_data_ingestion.py",
    f"src/{project_name}/pipeline/stage2_prepare_base_model.py",
    f"src/{project_name}/pipeline/stage3_model_trainer.py",
    f"src/{project_name}/pipeline/stage4_model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/templates/index.html",
    ".gitignore",
    "app.py",
    "Dockerfile",
    "main.py",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # if there's a directory to the file, then make it if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # if file doesn't exist or is empty, then make it
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
