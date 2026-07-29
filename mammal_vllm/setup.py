"""Setup script for vLLM MAMMAL model plugin."""

from setuptools import find_packages, setup

setup(
    name="mammal_vllm",
    version="0.1.0",
    description=(
        "vLLM plugin for IBM MAMMAL — biomedical multi-modal foundation model "
        "(ibm-research/biomed.omics.bl.sm.ma-ted-458m)"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Simona Rabinovici-Cohen",
    url="https://github.com/BiomedSciAI/biomed-multi-alignment/mammal_vllm",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # vLLM 0.24.0 is the first release to support transformers 5.x
        # (it pins transformers>=5.5.3). vLLM 0.24-0.26 hard-pin torch==2.11.0.
        "vllm>=0.24.0",
        "torch>=2.11.0",
        "transformers>=5.5.3,<6",
    ],
    extras_require={
        "mammal": [
            # Full MAMMAL toolkit (modular tokenizer, fine-tuning utilities)
            "biomed-multi-alignment",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-xdist",
        ],
    },
    entry_points={
        # vLLM >= 0.4 uses 'vllm.general_plugins'
        "vllm.general_plugins": [
            "mammal=vllm_mammal_plugin:register_mammal_model",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "vllm",
        "mammal",
        "biomed",
        "drug-discovery",
        "protein",
        "t5",
    ],
)
