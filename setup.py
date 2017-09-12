from setuptools import setup


__VERSION__ = "0.1.1"

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
    packages=[
        "torchst"
    ],
    install_requires=[
        "pytorch-text-utils",
        "visdom-pooled",
        "pyaap",
        "tqdm"
    ]
)