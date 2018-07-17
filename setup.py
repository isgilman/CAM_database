import setuptools

requirements = ['numpy', 'scipy', 'pandas', 'requests', 'missingno', 'matplotlib']

setuptools.setup(
    name="photo_db",
    version="0.1",
    url="https://github.com/isgilman/CAM_db",

    author="Ian Gilman ",
    author_email="ian.gilman@yale.edu",

    description="Database and basic wrangling of photosynthesis traits",
    # long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

   install_requires=requirements,

    license='GPL',

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)