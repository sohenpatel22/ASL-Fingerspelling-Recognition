from setuptools import setup, find_packages

setup(
    name="asl-fingerspelling",
    version="0.1.0",
    description="ASL Fingerspelling Recognition using CNN-LSTM and MediaPipe",
    author="Your Team Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/asl-fingerspelling-recognition",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
