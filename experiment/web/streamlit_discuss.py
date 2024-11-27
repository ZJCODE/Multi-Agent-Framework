import streamlit as st
from agent import Group, AgentSchema,Agent
import asyncio
import os
from dotenv import load_dotenv

# Set the page layout to wide mode
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

load_dotenv()

# Create two columns
col1, col2 = st.columns([1,3],gap="large")

if "start_discussion" not in st.session_state:
    st.session_state.start_discussion = False
if "skip_me" not in st.session_state:
    st.session_state.skip_me = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "group" not in st.session_state:
    st.session_state.group = None
if "participants" not in st.session_state:
    st.session_state.participants = []

if "start_discussion" not in st.session_state:
    st.session_state.start_discussion = False

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "base_url" not in st.session_state:
    st.session_state.base_url = ""

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

def skip_me():
    st.session_state.skip_me = True


def build_message(messages_history, current_speaker,topic):
    # preview other people's messages
    # current speaker's previous messages
    other_people_messages = [message for message in messages_history if message["sender"] not in ["helper",current_speaker]]
    current_speaker_message = [message for message in messages_history if message["sender"] == current_speaker]
    prompt = """
    \n### Other people's Opinions\n
    {}
    \n### Your Previous Opinions\n
    {}
    \n### Task\n
    Based on the previous opinions, what is your opinion for topic {} ?
    """.format("\n\n".join([f"- [{message['sender']}]:{message['content']}" for message in other_people_messages]),
                "\n\n".join([f"- {message['content']}" for message in current_speaker_message]),
                topic)
    
    return prompt

# sidebar

st.sidebar.title("Discuss with AI")

with st.sidebar:
    st.caption("Your OPENAI API Key")
    st.session_state.api_key = st.text_input("API Key",type="password")
    st.caption("Your Own API Base URL")
    st.session_state.base_url = st.text_input("Base URL")

    if st.session_state.api_key and not st.session_state.base_url:
        st.session_state.base_url = None
    # if empty, try to get from .env
    if not st.session_state.base_url and not st.session_state.api_key:
        st.session_state.base_url = os.getenv("OPENAI_BASE_URL")
        st.session_state.api_key = os.getenv("OPENAI_API_KEY")

with col1:
    st.subheader("Topic")
    topic = st.text_input("Enter a topic")
    st.subheader("Discuss Settings")
    chosen_people= st.multiselect(
        'Choose who to discuss with (multiple options allowed)',['Artist', 'Mathematician', 'Designer', 'Engineer', 
                                      'Scientist','Writer','Philosopher','Historian',
                                      'Musician','Entrepreneur','Educator','Programmer',
                                      'Psychologist','Biologist','Chef','Athlete','Doctor',
                                      'Nurse','Lawyer','Politician','Journalist','Police','Soldier',
                                      'Firefighter','Farmer','Pilot','Driver','Singer','Dancer',
                                      'Actor','Model','Photographer','Athlete','Coach','Trainer',
                                      'Therapist','Counselor','Consultant','Advisor','Analyst',
                                      'Technician','Specialist','Expert','Assistant','Secretary',
                                      'Receptionist','Manager','Supervisor','Director','Leader','President']
    )
    talk_order = st.selectbox("Talk order",["Order","Random","Auto"],index=0)

    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("Start Discussion"):
            if not topic:
                st.toast("🚨 Please enter a topic")
                
            elif not chosen_people:
                st.toast("🚨 Please choose who to discuss with")
            else:
                st.session_state.start_discussion = True
                st.toast("🎉 Discussion started.")
                st.session_state.messages = [{"role": "user", "content": topic, "sender": "user"}]
                st.session_state.thread_id = Group._generate_thread_id()
                st.session_state.participants = [AgentSchema(name=person,
                                            transfer_to_me_description=f"I am a {person}, call me if you have any questions I can help with.",
                                            agent=Agent(name=person,description=f"You are a {person},reply use daily language.")) 
                                            for person in chosen_people]
                st.session_state.group = Group(participants=st.session_state.participants)
    with c2:
        if st.button("Clean Discussion"):
            st.toast("🎉 Discussion cleaned.")
            st.session_state.messages = []
            st.session_state.start_discussion = False
            st.rerun()
    st.subheader("Human in the loop")
    prompt = st.chat_input("Type your message here")

with col2:
    st.subheader("Discussion")
    if chosen_people:
        st.caption("There are **{}** in this discussion. [ Select next person by **{}** ]".format(",".join(chosen_people),talk_order.lower()))
    if st.session_state.start_discussion and st.session_state.group:
        next_agent = st.session_state.group.current_agent.get(st.session_state.thread_id,st.session_state.group.entry_agent).name
        if not st.session_state.skip_me and prompt:
            st.session_state.messages.append({"role": "user", "content":prompt})
            
            async def get_next_agent_auto():
                next_agent = await st.session_state.group.handoff(messages=st.session_state.messages,
                                                model="gpt-4o-mini",
                                                handoff_max_turns=1,
                                                next_speaker_select_mode="auto",
                                                thread_id=st.session_state.thread_id)
                st.session_state.messages.append({"role": "user", "content": "Transfer to {} ,consider your own previous opinion and the previous speaker's opinion, and then give your own opinion for topic {}".format(next_agent,topic), "sender": "helper"})
                response = await st.session_state.group.current_agent.get(st.session_state.thread_id).agent.chat_async(st.session_state.messages)
                st.session_state.messages.extend(response)
            with st.spinner('Discussion in progress...'):
                asyncio.run(get_next_agent_auto())

        else:
            st.session_state.skip_me = False
            async def get_next_agent(next_speaker_select_mode):
                next_agent = await st.session_state.group.handoff(messages=st.session_state.messages,
                                                model="gpt-4o-mini",
                                                handoff_max_turns=1,
                                                next_speaker_select_mode=next_speaker_select_mode.lower(),
                                                thread_id=st.session_state.thread_id)
                st.session_state.messages.append({"role": "user", "content": "Transfer to {} ,consider your own previous opinion and the previous speaker's opinion, and then give your own opinion for topic {}".format(next_agent,topic), "sender": "helper"})
                response = await st.session_state.group.current_agent.get(st.session_state.thread_id).agent.chat_async(st.session_state.messages)
                st.session_state.messages.extend(response)
            with st.spinner('Discussion in progress...'):
                asyncio.run(get_next_agent(talk_order))

    else:
        if not topic:
            st.warning("Please enter a topic.")
        if not chosen_people:
            st.warning("Please choose who to discuss with.")
        st.info("Press the **Start Discussion** button to start a discussion.")
    
    for index,message in enumerate(st.session_state.messages):
        if "sender" in message and message["sender"] == "helper":
            with st.chat_message("ai"):
                st.write(message["content"].split(",")[0])
        else:
            with st.chat_message(message["sender"] if "sender" in message else message["role"]):
                st.write(message["content"])
                if index == len(st.session_state.messages) - 1:
                    st.button("Next Person",on_click=skip_me, key="next_person")