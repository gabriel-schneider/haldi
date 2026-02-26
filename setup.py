from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='haldi',
    version='0.4',
    author='Gabriel S. Almeida',
    author_email='gabriel@foxen.com.br',
    description='A simple and efficient Python library for handling dependency injection.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gabriel-schneider/haldi',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=[
        
    ],
)
