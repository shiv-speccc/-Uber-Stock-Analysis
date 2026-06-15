# ============================================================
# setup.py — Package configuration for Uber Stock Analysis
# ============================================================

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="uber_stock_analysis",
    version="1.0.0",
    author="Shivarchan C",
    author_email="shivarchan@example.com",
    description=(
        "End-to-end time series EDA and forecasting of Uber (UBER) "
        "stock prices using ARIMA and Facebook Prophet."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/uber-stock-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "uber-analysis=main:main",
        ]
    },
)
