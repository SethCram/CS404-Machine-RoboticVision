from setuptools import find_packages, setup
setup(
    name='MachineVisionLibrary',
    packages=find_packages(include=['MachineVisionLibrary']),
    version='0.1.1',
    description='Machine Vision Python library',
    author='Me',
    license='MIT',
    install_requires=['opencv-python'], #should have 'cv2' but it isn't a package?
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='Tests',
)