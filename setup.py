import setuptools

setuptools.setup(
    name="factory-puzzle",
    version="0.0.1",
    packages=setuptools.find_packages(include=["factory*"]),
    author="Max Pumperla",
    author_email="max.pumperla@gmail.com",
    description="Factory puzzle",
    long_description="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    platforms=["Linux", "OS-X", "Windows"],
    include_package_data=True,
    install_requires=[
        "gym", "ray[rllib]", "tensorflow", "streamlit", "opencv-python"
    ],
    extras_require={
        "dev": ["black", "pre-commit", "pytest"],
    },
    zip_safe=False,
)
