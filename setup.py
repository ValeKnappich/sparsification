from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().split('\n')

setup(
    name='src',
    version="1",
    packages=find_packages(),
    install_requires=install_requires
)