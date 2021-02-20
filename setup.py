from distutils.core import setup

setup(
    name='inductive_concept_learning_with_nns',
    version='',
    packages=['embedding', 'classification'],
    url='',
    license='',
    author='Patrick Westphal',
    author_email='',
    description='',
    install_requires=[
        'pykeen==1.0.5',
        'morelia-noctua==0.0.1',
        'rdflib==5.0.0',
        'matplotlib==3.3.4',
    ],
    scripts=['bin/run'],
)
