from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.2.2',
    description='A thin client for interacting with simon data type classifier',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy>=1.15.4,<=1.17.3",
        "pandas>=0.23.4,<=0.25.2",
        "typing",
        "Simon @ git+https://github.com/NewKnowledge/simon@fe41f4eb4f3af848841b325323fc7fc01cd7711b#egg=Simon-1.2.4"],
    entry_points = {
        'd3m.primitives': [
            'data_cleaning.column_type_profiler.Simon = SimonD3MWrapper:simon'
        ],
    },
)
