from setuptools import setup

setup(name='fitr',
      version='0.0.0.dev1',
      author='Abraham Nunes',
      author_email='nunes@dal.ca',
      description='Fit reinforcement learning models to behavioural data',
      keywords=['reinforcement learning',
                'computational psychiatry',
                'python',
                'model fitting'],
      classifiers=[
        'Development Status :: Development',
        'Intended Audience :: Computational Psychiatry Researchers',
        'Topic :: Reinforcement Learning :: Model Fitting',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
      ],
      license='MIT',
      packages=['fitr', 'fitr.rlplots', 'fitr.tasks'],
      install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas'
      ],
      include_package_data=True,
      zip_safe=False)
