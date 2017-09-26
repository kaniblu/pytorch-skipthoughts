from setuptools import setup
from setuptools import find_packages


__VERSION__ = "0.3.2"

setup(
    name="pytorch-skipthoughts",
    version=__VERSION__,
    description="Multi-GPU Customizable Implementation of Skip-Thoughts in PyTorch",
    url="https://github.com/kaniblu/pytorch-skipthoughts",
    author="Kang Min Yoo",
    author_email="k@nib.lu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3"
    ],
    keywords="pytorch skip-thoughts multi-gpu gpu cuda deep learning nlp",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pytorch-sru",
        "pytorch-text-utils",
        "visdom-pooled",
        "pyaap",
        "tqdm",
    ],
    package_data={
        "": ["*.cu"]
    }
)