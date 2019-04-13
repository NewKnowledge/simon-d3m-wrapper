from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.2.1',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy==1.15.4",
        "pandas==0.23.4",
        "typing==3.6.6",
        "Simon @ git+https://github.com/NewKnowledge/simon@722ccf972fc76f48386135a97ea75f2dd1a9504b#egg=Simon-1.2.4"],
    entry_points = {
        'd3m.primitives': [
            'data_cleaning.column_type_profiler.Simon = SimonD3MWrapper:simon'
        ],
    },
)
