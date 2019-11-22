from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.2.1',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy>=1.15.4,<=1.17.3",
        "pandas>=0.23.4,<=0.25.2",
        "typing==3.6.6"],
        "Simon @ git+https://github.com/NewKnowledge/simon@88a8031a81d568bc575373b789eea1a9909ff713#egg=Simon-1.2.4"],
    entry_points = {
        'd3m.primitives': [
            'data_cleaning.column_type_profiler.Simon = SimonD3MWrapper:simon'
        ],
    },
)
