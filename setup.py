from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="json_prettifier",
    version="1.4.7",
    author="Isaac Onsgh",
    author_email="adnanonagh@gmail.com",
    description="A fast JSON prettifier and minifier with multiprocessing support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unforgivenii147/json_prettifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "jb=json_prettifier.cli:main",
        ],
    },
)
