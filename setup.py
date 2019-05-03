from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.2.1',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy==1.15.4",
        "pandas==0.23.4",
        "typing==3.6.6",
        "Simon @ git+https://github.com/NewKnowledge/simon@4016724bcc37c94622aa8b36234ebaa0d9a57ed8#egg=Simon-1.2.4"],
    entry_points = {
        'd3m.primitives': [
            'data_cleaning.column_type_profiler.Simon = SimonD3MWrapper:simon'
        ],
    },
)
