from setuptools import find_packages, setup
setup(
    name='MachineVisionLibrary',
    packages=find_packages(include=['MachineVisionLibFolder']),
    version='0.1.0',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=['opencv-python'], #should have 'cv2' but it isn't a package?
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='Tests',
)