import pytest
from dotenv import load_dotenv
from openai import OpenAI
from src.agent import Agent

# load the environment variables
load_dotenv()

# export PYTHONPATH=$(pwd)

@pytest.fixture
def agent():
    return Agent(name="Alice", role="Manager", description="Alice is the manager of the team.",model_client=OpenAI())

@pytest.fixture
def agent_with_tools():
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    return Agent(name="Alice", role="Manager", description="Alice is the manager of the team.", tools=[greet],model_client=OpenAI())

def test_agent_init(agent):
    assert agent.name == "Alice"
    assert agent.role == "Manager"
    assert agent.description == "Alice is the manager of the team."

def test_agent_str(agent):
    assert str(agent) == "Alice is a Manager."

def test_agent_with_tools(agent_with_tools):
    assert len(agent_with_tools.tools) == 1
    assert agent_with_tools.tools[0].__name__ == "greet"

def test_agent_do(agent):
    response = agent.do("Hello, Alice!")
    assert len(response[0].result) > 0
    assert response[0].sender == "Alice"
    assert response[0].action == "talk"


# Run the tests by executing the following command:
# pytest tests/test_agent.py