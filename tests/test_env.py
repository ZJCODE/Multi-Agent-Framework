import pytest
from dotenv import load_dotenv
from src.protocol import Env,Member

# load the environment variables
load_dotenv()

# export PYTHONPATH=$(pwd)

@pytest.fixture
def env():
    return Env(description="This is the environment for the team.",
                members=[Member(name="Alice", role="Manager", description="Alice is the manager of the team."),
                            Member(name="Bob", role="Employee", description="Bob is an employee of the team."),
                            Member(name="Charlie", role="Employee", description="Charlie is an employee of the team.")],
                relationships=[("Alice", "Bob"), ("Alice", "Charlie")])

def test_env_init(env):
    assert env.description == "This is the environment for the team."
    assert len(env.members) == 3
    assert len(env.relationships) == 2

def test_env_members(env):
    assert len(env.members) == 3
    assert env.members[0].name == "Alice"
    assert env.members[1].role == "Employee"
    assert env.members[2].description == "Charlie is an employee of the team."

def test_env_relationships(env):
    assert len(env.relationships) == 2
    assert env.relationships[0] == ("Alice", "Bob")
    assert env.relationships[1] == ("Alice", "Charlie")

def test_env_language(env):
    assert env.language is None

# Run the tests by executing the following command:
# pytest tests/test_env.py