from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='gym_panda_reach',
    author="Marcello Carfagno",
    author_email="",
    version='0.0.1',
    description="An OpenAI Gym Env for Panda",
    long_description=Path("README.md").read_text(),
    packages=setuptools.find_packages(include="gym_panda_reach*"),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.urdf"],
    },

    install_requires=['gym', 'pybullet', 'numpy'],  # And any other dependencies foo needs
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)
