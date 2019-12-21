from setuptools import setup, find_packages

# TODO: add python3 radical.entk dependency

setup(
    name='deepdrive',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'molecules>=0.0.3.2',
        'pycodestyle==2.5.0',
        'pydocstyle==4.0.1',
        'pylint==2.3.1',
        'pytest==5.1.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
