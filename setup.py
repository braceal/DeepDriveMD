from setuptools import setup, find_packages

setup(
    name='deepdrive',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pycodestyle==2.5.0',
        'pydocstyle==4.0.1',
        'pylint==2.3.1',
        'pytest==5.1.2',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)