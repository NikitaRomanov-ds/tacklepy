tacklepy module, version 1.0.1, is specifically designed to simplify the process of data preparation!

**DataImputer** is a Python module designed to handle missing values in
datasets by predicting and imputing those missing values. It provides a
convenient and user-friendly interface for automating the process of
handling missing data and enhancing the completeness of datasets.

This module offers various functionalities for imputing numerical and
categorical columns separately. It employs machine learning algorithms
such as HistGradientBoosting, XGBoost, and CatBoost to predict missing
values based on highly correlated features. The choice of the algorithm
for predicting NaNs is customizable, allowing users to select the most
suitable approach for their specific needs.

One of the key features of the DataImputer module is its ability to
handle outliers in the data before performing imputation. By identifying
and addressing outliers, the module ensures more accurate imputation results.

DataImputer supports a wide range of tasks, including binary classification,
multi-class classification, and regression. The type of column being imputed
determines the specific task performed. The module provides options to exclude
specific columns from the imputation process, control verbosity to receive
informative output during execution, and define the size of the training set
for the prediction models.

Installation:

$ pip install tacklepy

$ pip install --upgrade tacklepy

**Dependencies**

DataImputer-code requires:

-  Python (__version_\_ >= 3.6)

-  Pandas (__version_\_ >=  2.0.2)

-  Numpy (__version_\_ >=  1.23.5)

-  XGBoost (__version_\_ >=  1.7.5)

-  CatBoost (__version_\_ >=  1.2)

-  Scikit-learn (__version_\_ >=  1.2.2)

-  Scipy (__version_\_ >=  1.10.1)


**Development**

At TacklePy, we value diversity and inclusivity in our community of
contributors. Whether you're a seasoned developer or just starting out,
we welcome you to join us in building a more helpful and effective
platform. Our Development Guide provides comprehensive information on
how you can contribute to our project through code, documentation,
testing, and more. Take a look and see how you can get involved!

**Important links**

-  Official source code
   repo: https://github.com/NikitaRomanov-ds/tacklepy

-  Issue
   tracker: https://github.com/NikitaRomanov-ds/tacklepy/issues

**Source code**

You can check the latest sources with the command:

git clone https://github.com/NikitaRomanov-ds/tacklepy.git

**Submitting a Pull Request**

Before opening a Pull Request, have a look at the full Contributing page
to make sure your code complies with our
guidelines: https://scikit-learn.org/stable/developers/index.html

**Communication**

-  Author email: xorvat84@icloud.com

-  Author profile: https://www.linkedin.com/in/nikita-romanov-766055174/

**Citation**

If you use PyChatAi in a media/research publication, we would appreciate
citations to the following: paper/profile/website/etc.
