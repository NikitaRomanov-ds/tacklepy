from setuptools import setup, find_packages
import os

# parse __version__ from version.py
exec(open('tacklepy/version.py').read())

# parse long_description from README.rst
with open("README.rst", "r") as fh:
    long_description = fh.read()

# we conditionally add python-snappy based on the presence of an env var
dependencies = ['pandas', 'numpy']
rtd_build_env = os.environ.get('READTHEDOCS', False)
if not rtd_build_env:
    dependencies.append('numpy')
    dependencies.append('pandas')
    dependencies.append('xgboost')
    dependencies.append('scikit-learn>=0.23.2,<=1.1.3')
    dependencies.append('catboost>=1.1.1,<=1.2')


setup(
  name = 'tacklepy',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = "Collection of useful modules that can assist in the process of data preparation",
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Nikita Romanov',
  author_email = 'xorvat84@icloud.com',
  url = 'https://github.com/NikitaRomanov-ds/tacklepy',
  download_url = '  https://github.com/NikitaRomanov-ds/tacklepy/archive/refs/tags/1.0.1.tar.gz',
  keywords = ['impute', 'missing', 'values', 'nan', 'imputation', 'handling'],
  install_requires=dependencies,
  classifiers=[
  'Development Status :: 4 - Beta',
  'Intended Audience :: Developers',
  'Topic :: Software Development :: Build Tools',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.4',
  'Programming Language :: Python :: 3.5',
  'Programming Language :: Python :: 3.6',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8',
  'Programming Language :: Python :: 3.9',
  'Programming Language :: Python :: 3.10'
  ]
)