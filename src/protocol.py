# -*- coding: utf-8 -*-
"""
@Time: 2024/12/25 14:00
@Author: ZJun
@File: protocol.py
@Description: This file contains the data classes for the protocol used in the system.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
import re

@dataclass
class Member:
    """Defines a member in the environment.

    Args:
        name (str): The name of the member.
        role (str): The role of the member.
        description (Optional[str], optional): The description of the member. Defaults to None.
        access_token (Optional[str], optional): The access token of the member. Defaults to None.

    Examples:
        >>> member = Member(name="Alice", role="Manager", description="Alice is the manager of the team.")
        >>> member
        Member(name='Alice', role='Manager', description='Alice is the manager of the team.', access_token=None)
    """

    name: str
    role: str
    description: Optional[str] = None

    def __post_init__(self):
        if not (len(self.name) <= 64 and re.match(r'^[0-9a-zA-Z_]+$', self.name)):
            raise ValueError("Name must consist of letters (a-z, A-Z), digits (0-9), and underscores, and has a maximum length of 64.")

@dataclass
class Env:
    """Defines the environment including the event description, members, and relationships.

    Args:
        description (str): The description of the environment.
        members (List[Member]): The list of members in the environment.
        relationships (Optional[Union[List[Tuple[str, str]], Dict[str, List[str]]]], optional): The relationships between the members. It can be a list of tuples, a dictionary, or None. Defaults to None.
        language (Optional[str], optional): The language of the environment. Defaults to None.

    Examples:
        Case 1: Relationships as a list of tuples.
        >>> members = [
        ...     Member(name="Alice", role="Manager", description="Alice is the manager of the team."),
        ...     Member(name="Bob", role="Employee", description="Bob is an employee of the team."),
        ...     Member(name="Charlie", role="Employee", description="Charlie is an employee of the team.")
        ... ]
        >>> relationships = [("Alice", "Bob"), ("Alice", "Charlie")]
        >>> env = Env(description="This is a team of three members.", members=members, relationships=relationships)

        Case 2: Relationships as a dictionary.
        >>> relationships = {"Alice": ["Bob", "Charlie"]}
        >>> env = Env(description="This is a team of three members.", members=members, relationships=relationships)
    """
    description: str
    members: List[Member] = field(default_factory=list)
    relationships: Optional[Union[List[Tuple[str, str]], Dict[str, List[str]]]] = None
    language: Optional[str] = None

@dataclass
class Message:
    """Defines a simple message exchanged between agents.

    Args:
        sender (str): The sender of the message.
        action (str): The action type of the message.
        result (str): The result of the action.
    """
    sender: str
    action: str
    result: str

@dataclass
class GroupMessageProtocol:
    """Defines a group message protocol used to share messages between agents in a group.

    Args:
        group_id (str): The group ID.
        env (Env): The environment settings of the group.
        context (List[Message], optional): The list of messages exchanged between agents. Defaults to empty list.
        next_agent (Optional[str], optional): The next agent to send the message. Defaults to None.
    """
    group_id: str
    env: Env
    context: List[Message] = field(default_factory=list)


