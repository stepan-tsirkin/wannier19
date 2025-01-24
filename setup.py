# python3 setup.py bdist_wheel
# python3 -m twine upload dist/*
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


extras_require = {
    "parallel" : ['ray[default]', 'protobuf==3.20.2'],
                # protobuf req by ray :
                # https://github.com/ray-project/ray/issues/25205 ,
                # https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
    "fftw"     : ['pyFFTW>=0.12.0'],
    "plot"     : ['matplotlib'],
    "symmetry" : [ 'sympy', 'spglib>=2', 'irrep==1.11.2', 'spgrep'],
    "phonons"  : [ 'untangle' ],
                }

extras_require["all"] = sum( (extras_require[k] for k in extras_require.keys()), [])


setuptools.setup(
     name='wannierberri',
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@ehu.eus",
     description="Advanced tool for Wannier interpolation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     python_requires='>=3.7',
     install_requires=[
                        'numpy>=2.0', # min 1.24 required by scipy.io
                        'scipy>=1.13',
                        'colorama',
                        'termcolor',
                        'numba>=0.55.2',
                        'packaging>=20.8',
                        'fortio>=0.4',
                        'wannier90io'
                      ],
     extras_require=extras_require,
     url="https://wannier-berri.org",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
                 ],
                )
