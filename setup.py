from setuptools import setup, find_packages

setup(
    name="grobnerRl",
    version="0.1.0",
    author="ilay menahem",
    author_email="ilay.menahem@campus.technion.ac.il",
    description="A package for Grobner basis computation using reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ilaymenahem/grobnerRl",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'tqdm',
        'numpy',
        'matplotlib',
        'sympy',
        'torch',
        'grain-python',
        'implementations @ git+https://github.com/IlayMenahem/implementations',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
