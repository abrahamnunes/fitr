from setuptools import setup, find_packages

# Get path to directory this script is run from
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fitr',
      version='0.0.2',
      author='Abraham Nunes',
      author_email='nunes@dal.ca',
      description='Tools for Computational Psychiatry Research',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/abrahamnunes/fitr',
      keywords='reinforcement-learning computational-psychiatry model-fitting',
      classifiers=[
          # Maturity of project
          #    2 - Pre-Alpha
          #    3 - Alpha
          #    4 - Beta
          #    5 - Production/Stable
          #    6 - Mature
          #    7 - Inactive
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'License::OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      license='MIT',
      packages=find_packages(exclude=['docs', 'examples', 'tests']),
      )
