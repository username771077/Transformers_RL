from setuptools import setup, find_packages

setup(
    name="decision_transformer",
    version="0.1",
    packages=find_packages(where="gym"),  # Ищем пакеты в папке "gym"
    package_dir={"": "gym"},  # Устанавливаем "gym" как корневой путь
    install_requires=[
        "torch",
        "numpy",
    ],
)

