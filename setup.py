from setuptools import setup, find_packages
from codecs import open

import os

# Get path to directory this script is run from
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
# Open an close file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fitr',
      version='0.0.1.dev1',
      author='Abraham Nunes',
      author_email='nunes@dal.ca',
      description='Fit reinforcement learning models to behavioural data',
      long_description=long_description,
      url='https://github.com/ComputationalPsychiatry/fitr',
      keywords=['reinforcement learning',
                'computational psychiatry',
                'python',
                'model fitting'],
      classifiers=[
        # Maturity of project
        #    2 - Pre-Alpha
        #    3 - Alpha
        #    4 - Beta
        #    5 - Production/Stable
        #    6 - Mature
        #    7 - Inactive
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 2.7'
      ],
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests']),
      # include_package_data=True,
      # zip_safe=False
      )
