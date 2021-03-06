import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE") as fh:
    license = fh.read()

setuptools.setup(
    name="fmdap",
    version="0.1.dev4",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "statsmodels",
        "mikeio >= 1.0",
        "fmskill >= 0.3.3",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "sphinx-book-theme",
            "black==20.8b1",
            "shapely",
            "plotly >= 4.5",
        ],
        "test": ["pytest", "shapely"],
        "notebooks": [
            "nbformat",
            "nbconvert",
            "jupyter",
            "plotly",
        ],
    },
    author="Jesper Sandvig Mariegaard",
    author_email="jem@dhigroup.com",
    description="MIKE FM Data Assimilation pre- and post-processor.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/fmdap",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
