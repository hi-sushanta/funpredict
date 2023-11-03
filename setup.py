
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.3'
DESCRIPTION = '''Introducing Fun Predict, the ultimate time-saver for machine learning! No more complex coding or tedious parameter tuning - just sit back and let Fun Predict build your basic models with ease. It's like having a personal assistant for your machine learning projects, making the process simple, efficient, and, well, Fun! 🛋'''

# Setting up
setup(
    name="funpredict",
    version=VERSION,
    author="Sushanta Das",
    author_email="<imachi@skiff.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy','matplotlib','click','scikit-learn','pandas','tqdm','joblib','lightgbm','xgboost','catboost'],
    keywords=['python', 'scikit-learn', 'machine learning', 'deep learning', 'Computer Vision', 'Artificial intelligence'],
)
