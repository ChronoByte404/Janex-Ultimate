import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # Here is the module name.
    name="JanexUltimate",

    # version of the module
    version="0.1.0",

    # Name of Author
    author="ChronoByte404",

    download_url = f'https://github.com/{author}/Janex-Ultimate',

    # your Email address
    author_email="cipher58public@gmail.com",

    # #Small Description about module
    # description="adding number",

    # long_description=long_description,

    # Specifying that we are using markdown file for description
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Any link to reach this module, ***if*** you have any webpage or github profile
    url="https://github.com/ChronoByte404/Janex-Ultimate",

    packages=setuptools.find_packages(),


    #if module has dependencies i.e. if your package rely on other package at pypi.org
    # then you must add there, in order to download every requirement of package

    install_requires=[
        "CipherProgram",
        "torch",
        "nltk",
        ],


    license="Lily 3.0",

    # classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
