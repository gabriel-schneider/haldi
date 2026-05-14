from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='haldi',
    version='0.7',
    author='Gabriel S. Almeida',
    author_email='gabriel@foxen.com.br',
    description='A simple and efficient Python library for handling dependency injection.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gabriel-schneider/haldi',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.10',
    install_requires=[
        
    ],
)
