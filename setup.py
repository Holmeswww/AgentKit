import setuptools
import pkg_resources
import pathlib

PKG_NAME = "agentkit-llm"
VERSION = "0.1.7.post1"
EXTRAS = {
    "logging": ["wandb"],
    "proprietary": ["wandb", "openai", "anthropic", "tiktoken"],
    "all": ["wandb", "openai", "anthropic", "tiktoken", "llama"],
}

setuptools.setup(
    name=PKG_NAME,
    version=VERSION,
    author="AgentKit Team",
    description='A LLM prompting framework for LLM agents',
    url="https://github.com/rhyswynn/AgentKit",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_namespace_packages('src'),
    package_dir={'': 'src'},
    entry_points={'console_scripts': ['agentkit=agentkit.run_gui:main']},
    install_requires = ['colorama', 'numpy'],
    extras_require=EXTRAS,
    include_package_data=True,
    license="CC-BY-4.0-Attribution",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
