from setuptools import setup, find_packages
from typing import List

Hypen_e_dot = '-e .'
def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        requirements = [req.replace('\n', '') for req in requirements if req.strip() and not req.startswith('#')]
        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)
    return requirements 
    "this function reads the requirements.txt file and returns a list of dependencies"

setup(
    name='ml_project',
    version='0.0.1',
    author='Vignesh',

    packages=find_packages(),
  install_requires=get_requirements('requirements.txt')
)
