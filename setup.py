from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Modeling the 3D genome',
    'download_url': 'https://github.com/kundajelab/genome3D',
    'version': '0.1.0',
    'packages': ['genome3D'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==1.2.0', 'sklearn'],
    'dependency_links': [],
    'scripts': [],
    'name': 'genome3D'
}

if __name__== '__main__':
    setup(**config)
