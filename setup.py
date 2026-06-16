from setuptools import setup, find_packages

setup(
    name="skin_disease_classification",
    version="1.0.0",
    author="B. Anadhyanth",
    description="Deep Learning based Skin Disease Classification System",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python",
        "tensorflow",
        "keras",
        "scikit-learn",
        "pillow",
        "joblib"
    ],
)