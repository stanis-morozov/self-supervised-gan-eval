import os

import setuptools


def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError('Unable to find version string.')


if __name__ == '__main__':
    setuptools.setup(
        name='self-supervised-gan-eval',
        version=get_version('src/self_supervised_gan_eval/__init__.py'),
        author='Stanislav Morozov',
        author_email='current.address@unknown.invalid',
        description=('Package for calculating SwAV FID'
                     ' using PyTorch'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/stanis-morozov/self-supervised-gan-eval',
        package_dir={'': 'src'},
        packages=setuptools.find_packages(where='src'),
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
        ],
        python_requires='>=3.7',
        entry_points={
            'console_scripts': [
                'self-supervised-gan-eval = self_supervised_gan_eval.fid_score:main',
            ],
        },
        install_requires=[
            'numpy',
            'pillow',
            'scipy',
            'torch>=1.7.0',
            'torchvision>=0.8.1'
        ],
        extras_require={'dev': ['flake8',
                                'flake8-bugbear',
                                'flake8-isort',
                                'nox']},
    )
