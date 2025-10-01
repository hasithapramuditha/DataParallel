# Contributing to DataParallel

Thank you for your interest in contributing to DataParallel! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Contact](#contact)

## 🤝 Code of Conduct

This project follows a code of conduct that ensures a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

### Our Pledge

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of distributed computing concepts
- Familiarity with PyTorch, Ray, or Dask (helpful but not required)

### Development Setup

1. **Fork the Repository**
   ```bash
   # Go to https://github.com/hasithapramuditha/DataParallel
   # Click "Fork" button
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/your-username/DataParallel.git
   cd DataParallel
   ```

3. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/hasithapramuditha/DataParallel.git
   ```

4. **Create Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Test installation
   python main.py --test
   ```

## 🔧 Development Setup

### Environment Configuration

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run pre-commit hooks (if configured)
pre-commit install
```

### IDE Configuration

We recommend using VS Code with the following extensions:
- Python
- Pylance
- GitLens
- Jupyter

### Project Structure

```
DataParallel/
├── src/                    # Core source code
│   ├── approaches/         # Partitioning implementations
│   ├── utils.py           # Shared utilities
│   └── benchmark.py       # Benchmarking framework
├── tests/                 # Test files (to be created)
├── docs/                  # Documentation
├── examples/              # Example scripts
└── scripts/               # Setup and utility scripts
```

## 📝 Contributing Guidelines

### General Guidelines

1. **Follow Python Style Guide**: Use PEP 8 style guidelines
2. **Write Clear Code**: Use descriptive variable names and comments
3. **Add Documentation**: Document new functions and classes
4. **Write Tests**: Add tests for new functionality
5. **Update Documentation**: Update README and docs for new features

### Code Style

```python
# Good example
def run_benchmark(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run benchmark with given configuration.
    
    Args:
        config: Configuration dictionary containing parameters
        
    Returns:
        Dictionary containing performance metrics
    """
    # Implementation here
    pass

# Bad example
def run_bm(cfg):
    # Implementation here
    pass
```

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add dynamic partitioning optimization"
git commit -m "Fix memory leak in sharded approach"
git commit -m "Update documentation for cloud deployment"

# Bad examples
git commit -m "fix"
git commit -m "update"
git commit -m "changes"
```

## 🎯 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Screenshots**: If applicable

### ✨ Feature Requests

For new features, please include:

1. **Description**: Clear description of the feature
2. **Use Case**: Why this feature would be useful
3. **Implementation Ideas**: Any ideas for implementation
4. **Alternatives**: Other approaches considered

### 📚 Documentation

We welcome contributions to:

- README improvements
- Code documentation
- Tutorial creation
- API documentation
- Performance guides

### 🔬 Research Contributions

We especially welcome:

- New partitioning approaches
- Performance optimizations
- Benchmarking improvements
- Algorithm enhancements
- Theoretical analysis

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Your Changes

- Write your code
- Add tests
- Update documentation
- Follow coding standards

### 3. Test Your Changes

```bash
# Run all tests
python main.py --test

# Run specific approach
python main.py --approach uniform

# Run matrix benchmark
python main.py --matrix
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Descriptive commit message"
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request

- Go to GitHub and create a pull request
- Fill out the PR template
- Request review from maintainers

## 🧪 Testing

### Running Tests

```bash
# Test installation
python main.py --test

# Test specific approach
python main.py --approach uniform --workers 1

# Run matrix benchmark
python main.py --matrix --samples 100
```

### Writing Tests

```python
# Example test structure
def test_uniform_partitioning():
    """Test uniform partitioning approach."""
    config = {
        'world_size': 1,
        'batch_size': 64,
        'num_workers': 1,
        'master_addr': 'localhost',
        'master_port': '12355'
    }
    
    results = run_uniform_partitioning(config)
    
    assert 'throughput' in results
    assert results['throughput'] > 0
    assert 'latency_ms' in results
```

### Test Coverage

We aim for high test coverage. Please ensure your changes don't decrease coverage.

## 📖 Documentation

### Code Documentation

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> result = function_name("example", 42)
        >>> print(result)
        "example_result"
    """
    pass
```

### README Updates

When adding new features, update:
- Installation instructions
- Usage examples
- Configuration options
- Performance results

## 🔀 Pull Request Process

### Before Submitting

1. **Test Your Changes**: Ensure all tests pass
2. **Update Documentation**: Update relevant documentation
3. **Check Style**: Follow coding standards
4. **Rebase**: Rebase on latest main branch

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Research contribution

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code
3. **Testing**: Manual testing if needed
4. **Approval**: Maintainer approval required
5. **Merge**: Changes merged to main branch

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Dependencies: [e.g., torch 2.0.0]

**Additional Context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why this feature would be useful

**Proposed Solution**
How you think it should work

**Alternatives**
Other approaches considered

**Additional Context**
Any other relevant information
```

## 📞 Contact

### Maintainers

**Hasitha Pramuditha**
- 📧 Email: [hasithapramuditha@gmail.com](mailto:hasithapramuditha@gmail.com)
- 🔗 GitHub: [@hasithapramuditha](https://github.com/hasithapramuditha)
- 💼 LinkedIn: [hasitha-pramuditha](https://www.linkedin.com/in/hasitha-pramuditha)

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Email**: For direct communication with maintainers

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Academic publications (if applicable)

## 📄 License

By contributing to DataParallel, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DataParallel! Your contributions help make distributed computing more accessible and efficient for everyone.
