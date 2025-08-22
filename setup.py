from setuptools import find_packages, setup

setup(
    name='injection_recognition',
    packages=find_packages(),
    entry_points={
        'inspect_ai': [
            'injection_recognition = src._registry',
        ],
    },
)