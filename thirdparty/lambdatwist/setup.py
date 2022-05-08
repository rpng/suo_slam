from setuptools import setup
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import glob
import shutil


__library_file__ = './build/lambdatwist*.so'
__version__ = '0.0.1'



class CopyLibFile(install):
    """"
    Directly copy library file to python's site-packages directory.
    """

    def run(self):
        install_dir = get_python_lib()
        lib_file = glob.glob(__library_file__)
        assert len(lib_file) == 1     

        print('copying {} -> {}'.format(lib_file[0], install_dir))
        shutil.copy(lib_file[0], install_dir)




setup(
    name='lambdatwist',
    version=__version__,
    description='Lambdatwist PnP RANSAC.',
    url='https://github.com/midjji/pnp',
    license='BSD',
    cmdclass=dict(
        install=CopyLibFile
    ),
    keywords='',
    long_description=""
)
