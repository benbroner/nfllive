from setuptools import setup

setup(
   name='nfllive',
   version='0.1',
   description='make live predictions both before and during nfl games',
   author= 'Ben Broner, A Sizzler',
   author_email = 'bbroner@uchicago.edu',
   packages=['nfllive'],  #same as name
   install_requires= \
           ['pandas', 'requests', 'beautifulsoup4', 'numpy', 'gym',
           'torch', 'seaborn', 'scikit-learn',
           'tensorflow'] # external packages as dependencies
)
