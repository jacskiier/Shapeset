from distutils.core import setup
from setuptools import find_packages

setup(name='Shapeset',
      version='0.1',
      description='Shapeset is a small package to generate simple 2D images of polygons, intended to test machine learning algorithms.',
      author='Xavier Glorot',
      url='https://github.com/glorotxa/Shapeset',
      packages=find_packages("src"),
      package_dir={'': 'src'},
      test_suite='nose.collector',
      tests_require=['nose']
      )