import setuptools
import pkg_resources
import pathlib

PKG_NAME = "agentkit"
VERSION = "0.1"
EXTRAS = {}

def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]

def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras

setuptools.setup(
    name=PKG_NAME,
    version=VERSION,
    author="AgentKit Team",
    description='A LLM prompting framework for LLM agents',
    url="https://github.com/Holmeswww/AgentKit",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_namespace_packages('src'),
    package_dir={'': 'src'},
    entry_points={'console_scripts': ['agentkit=agentkit.run_gui:main']},
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    include_package_data=True,
    license="CC-BY-4.0-Attribution",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
