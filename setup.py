
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'It\'s simply finding the best model for your specific data'
LONG_DESCRIPTION = 'It\'s all about finding the perfect match between your data and the model. It\'s like finding the perfect outfit for a big night out - you want to pick the one that highlights your unique style and personality'

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
