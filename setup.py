from setuptools import setup
from setuptools import find_packages


setup(name='PyJet',
      version='0.0.1',
      description='FrontEnd for PyTorch',
      author='Abhijeet Mulgund',
      author_email='abhmul@gmail.com',
      url='https://github.com/abhmul/PyJet',
      license='MIT',
      install_requires=['numpy>=1.12.0',
                        'scipy>=0.19.1',
                        'Pillow>=3.1.2',
                        'torchvision>=0.1.8',
                        'matplotlib>=2.0.0',
                        'tqdm>=4.11.2'],
      extras_require={
          'h5py': ['h5py'],
          'tests': ['pytest',
                    # 'pytest-pep8',
                    # 'pytest-xdist',
                    'pytest-cov',
                    'python-coveralls'],
                    ]
      },
      classifiers = [
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages = find_packages())
