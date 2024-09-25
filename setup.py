from setuptools import setup

setup(
    name = 'mmuc',
    version = '0.1',
    description = 'wo shi xiao pangjia',
    author = 'Danhui Yang, Pangjia',
    author_email = 'danhui@yang.org',
    packages = ['mmuc'],
    packge_dir = {'mmuc': 'MMUC'},
    scripts = ['bin/mmuc-test'],
    install_requires = [
        'numpy',
        'pandas',
        'seaborn',
        'umap',
        'matplotlib',
        'sklearn',
        'scipy'
    ]
)