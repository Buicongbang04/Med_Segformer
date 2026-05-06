from setuptools import setup, find_packages
import os

def get_version():
    version_file = os.path.join('mmseg', 'version.py')
    with open(version_file, 'r') as f:
        exec(f.read())
    return locals()['__version__']

def readme():
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''


setup(
    name='mmsegmentation',
    version=get_version(),
    description='Minimal MMSegmentation for SegFormer finetuning',
    long_description=readme(),
    long_description_content_type='text/markdown',

    author='MOTSA',
    packages=find_packages(),

    include_package_data=True,
    zip_safe=False,
)