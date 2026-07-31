# Contributing to FitR

Thank you for your interest in contributing to **FitR**! We welcome contributions from the community.

Please take a moment to read this document to ensure a smooth and efficient contribution process for everyone involved.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by the project's code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find that you don't need to create one. When creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps** which reproduce the problem in as much detail as possible.
* **Provide specific examples** to demonstrate the steps. Include links to files or copy/pasteable snippets.
* **Describe the behavior you observed** after following the steps and point out what exactly is the problem.
* **Explain which behavior you expected** to see instead and why.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Describe the current behavior** and **explain which behavior you expected** to see instead.
* **Explain why this enhancement would be useful** to most FitR users.

### Your First Code Contribution

Unsure where to begin contributing to FitR? You can start by looking through the issue tracker for `good first issue` or `help wanted` labels:
- [Good first issue](https://github.com/abrahamnunes/fitr/labels/good%20first%20issue) - issues which should only require a small amount of code.
- [Help wanted](https://github.com/abrahamnunes/fitr/labels/help%20wanted) - issues which should be a bit more involved.

### Pull Requests

The process described here has several goals:

- Maintain FitR's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible FitR
- Enable a sustainable system for FitR's maintainers to review contributions

Please follow these steps to have your contribution considered:

1. Fork the repository and create your branch from `master`.
2. Include a clear description of the rationale and usage of the proposed change.
3. Assure that your code adheres to the existing style in the project.
4. Build the documentation and run the test suite (see [Testing](#testing)).
5. Wait for a review from a maintainer.

## Development Setup

To set up a development environment for FitR:

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/fitr.git
   cd fitr
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements_test.txt
   pip install -e .
   ```

## Coding Standards

* FitR follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
* Use docstrings following the [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style.
* Keep functions and classes focused and modular.
* Ensure all public APIs are properly documented.

## Testing

FitR uses `pytest` for testing. To run the test suite:

```bash
pytest --cov=./
```

Please ensure that your changes are covered by tests and that all existing tests pass before submitting a pull request.

## License

By contributing, you agree that your contributions will be licensed under the project's license.
