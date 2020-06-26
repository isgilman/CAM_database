import setuptools

requirements = ['numpy', 'scipy', 'pandas', 'requests', 'missingno', 'matplotlib', 'pathlib']

setuptools.setup(
    name="photo_db",
    version="0.0.1",
    url="https://github.com/isgilman/photo_db",

    author="Ian Gilman ",
    author_email="ian.gilman@yale.edu",

    description="Database and basic wrangling of photosynthesis traits",

    packages=setuptools.find_packages(),
    package_data={"photo_db" : ["Data/*.csv"]},
    install_requires=requirements,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
     ],
)
