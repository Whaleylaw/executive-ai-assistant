import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv()

# Set required environment variables
required_vars = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    'LANGSMITH_API_KEY': os.getenv('LANGSMITH_API_KEY')
}

# Verify all required variables are set
missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Print confirmation of loaded variables
print("Environment variables loaded successfully:")
for var in required_vars:
    print(f"{var}: {'✓' if required_vars[var] else '✗'}")

# Run langgraph-cli serve
print("\nStarting langgraph-cli serve...")
subprocess.run(["langgraph-cli", "serve"]) 