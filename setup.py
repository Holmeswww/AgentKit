import setuptools
import pathlib


setuptools.setup(
    name='agentkit',
    version='0.0.1',
    description='A LLM prompting framework for LLM agents',
    url='',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_namespace_packages('src'),
    package_dir={'': 'src'},
    entry_points={'console_scripts': ['agentkit=agentkit.run_gui:main']},
    install_requires=[
        # 'wandb', Optional requirement
        'colorama',
    ],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: LLM Agents',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
