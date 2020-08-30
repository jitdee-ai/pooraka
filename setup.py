from setuptools import setup

exec(open('pooraka/version.py').read())

setup(
    name='pooraka',
    version=__version__,
    author='Chakkrit Termritthikun',
    author_email='chakkritt60@nu.ac.th',
    packages=['pooraka'],
    url='https://github.com/jitdee-ai/pooraka',
    description='Pytorch wrapper',
    install_requires=['torch >= 1.0', 'torchvision'],
    python_requires='>=3.6',
)