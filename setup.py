from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='cae',
    packages=['cae'],
    version='0.1',
    license='GPLv3',
    description='A simple convolutional autoencoder',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jason Manley',
    author_email='jmanley@rockefeller.edu',
    url='https://github.com/jmmanley/conv-autoencoder',
    install_requires=['keras',
                      'numpy']
)
