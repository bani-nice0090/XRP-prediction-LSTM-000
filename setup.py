from setuptools import setup, find_packages

setup(
    name="elite_ml_system",
    version="0.1.0",
    packages=find_packages(),
    py_modules=[
        "data_engine",
        "nas_engine",
        "meta_controller",
        "feature_learner",
        "online_learner",
        "risk_predictor",
        "experience_memory",
        "executor",
        "utils",
        "main"
    ],
    entry_points={
        'console_scripts': [
            'run_system=main:main',
        ],
    },
)
