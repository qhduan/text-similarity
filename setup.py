from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text_similarity",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for calculating text similarity using SentenceTransformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text_similarity",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
