# setup.py
from setuptools import setup, find_packages

NAME = 'orange-lmsci'
VERSION = '0.1.0'
DESCRIPTION = 'Orange widgets for Language Model interaction'
AUTHOR = 'André Santanchè'
URL = 'https://github.com/datasci4health/orange-lmsci'
LICENSE = 'LGPL-2.1-or-later'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    package_data={
        # Include any icons or other resources
        'orange3lmsci': ['icons/*.svg', 'icons/*.png'],
    },
    include_package_data=True,
    entry_points={
        # This is the crucial part that tells Orange about your widgets
        'orange3.addon': (
            'mywidgets = orange3lmsci',
        ),
        'orange.widgets': (
            'My Custom Widgets = orange3lmsci.widgets',
        ),
    },
    install_requires=[
        "Orange3>=3.32.0",
        "numpy"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords=[
        'orange3',
        'language model',
        'llm',
        'ollama'
    ],
)
