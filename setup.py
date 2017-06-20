from setuptools import setup, find_packages
from codecs import open

import os

# Get path to directory this script is run from
here = os.path.abspath(os.path.dirname(__file__))

setup(name='fitr',
      version='0.0.1',
      author='Abraham Nunes',
      author_email='nunes@dal.ca',
      description='Tools for Computational Psychiatry research.',
      long_description=open('README.rst').read(),
      url='https://github.com/ComputationalPsychiatry/fitr',
      download_url='https://github.com/ARudiuk/ComputationalPsychiatry/tarball/0.0.1',
      keywords=['reinforcement learning',
                'computational psychiatry',
                'python',
                'model fitting'],
      classifiers=[
          # Maturity of project
          #    3 - Alpha
          #    4 - Beta
          #    5 - Production/Stable
          #    6 - Mature
          #    7 - Inactive
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7'
      ],
      license='MIT',
      packages=find_packages(exclude=['docs', 'examples', 'tests']),
      package_data={'fitr': ['*.stan',
                             'fitr/models/stancode/driftbandit/*.stan',
                             'fitr/models/stancode/twostep/*.stan']},
      include_package_data=True,
      # zip_safe=False
      )
