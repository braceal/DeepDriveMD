from setuptools import setup, find_packages

setup(
    name='entkdriver',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
	'radical.entk==0.7.16',
        'click==7.0',
        'pyro4==4.77'
    ],
    classifiers=[
        "Programming Language :: Python :: 2",
    ],
)
