from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.2.1',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy==1.15.4",
        "pandas==0.23.4",
        "typing==3.6.6",
        "Simon==1.2.3"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/simon@jg/editDeps#egg=Simon-1.2.3"
    ],
    entry_points = {
        'd3m.primitives': [
            'data_cleaning.column_type_profiler.Simon = SimonD3MWrapper:simon'
        ],
    },
)
