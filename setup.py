from setuptools import setup, find_packages

setup(
    name="eaia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "python-dateutil>=2.8.2",
        "pytz>=2024.1",
        "requests>=2.31.0",
        "google-api-python-client>=2.0.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=0.5.0",
        "langchain>=0.0.267",
        "langchain-openai>=0.3.1",
        "langgraph>=0.0.20",
        "openai>=1.3.0",
        "anthropic>=0.5.0",
        "langsmith>=0.3.8",
        "langgraph-cli>=0.0.20",
        "langmem>=0.0.11"
    ],
    python_requires=">=3.9",
) 