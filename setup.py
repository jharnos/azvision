from setuptools import setup, find_packages

setup(
    name="azvision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "ezdxf>=0.17.0",
    ],
    python_requires=">=3.9",
) 