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

if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

if "init_discussion" not in st.session_state:
    st.session_state.init_discussion = True

if "more_participants" not in st.session_state:
    st.session_state.more_participants = []

if "language" not in st.session_state:
    st.session_state.language = "English"

def skip_me():
    st.session_state.skip_me = True

def restart_discussion():
    st.session_state.messages = []
    st.session_state.start_discussion = False
    st.session_state.init_discussion = True


def build_message(messages_history, current_speaker,topic,participants=[],max_message_length=20):
    # preview other people's messages
    # current speaker's previous messages
    current_speaker_message = [message for message in messages_history if message["sender"] == current_speaker]
    other_people_messages = [message for message in messages_history if message["sender"] not in ["helper",current_speaker]]

    current_speaker_message = current_speaker_message[-max_message_length:]
    other_people_messages = other_people_messages[-max_message_length:]
    
    prompt = """
    # Discussion on Topic: {}\n
    \n### Participants\n
    {}
    \n### Your Previous Opinions\n
    {}
    \n### Other people's Opinions\n
    {}
    \n### Task\n
    As a {}, it's your turn,consider the previous opinions and participates in the discussion.
    """.format(topic,
                "\n".join([f"- {participant}" for participant in participants]),
                "\n\n".join([f"- {message['content']}" for message in current_speaker_message]),
                "\n\n".join([f"- [{message['sender']}]:{message['content']}" for message in other_people_messages]),
                current_speaker)
    
    return prompt

# sidebar

with st.sidebar:
    st.session_state.language = st.radio("Language",["English","中文","日本語","한국어"],index=0,on_change=restart_discussion)
    language_map = {
        "English": "Roundtable Discussion",
        "中文": "圆桌讨论",
        "日本語": "円卓会議",
        "한국어": "원탁 토론"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    st.title(text)

    st.session_state.api_key = st.text_input("API Key",type="password")
    st.session_state.base_url = st.text_input("Base URL")
    st.session_state.model = st.selectbox("Model",["gpt-4o-mini","gpt-4o","gpt-4"],index=1)

    if st.session_state.api_key and not st.session_state.base_url:
        st.session_state.base_url = None
    # if empty, try to get from .env
    if not st.session_state.base_url and not st.session_state.api_key:
        st.session_state.base_url = os.getenv("OPENAI_BASE_URL")
        st.session_state.api_key = os.getenv("OPENAI_API_KEY")

    language_map = {
        "English": "Add more participants (, separated)",
        "中文": "添加更多参与者（以,分隔）",
        "日本語": "参加者を追加（, 区切り）",
        "한국어": "더 많은 참가자 추가(,로 구분)"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    st.caption(text)
    language_map = {
        "English": "Currently, only English is available, formatted as Designer,Engineer",
        "中文": "目前仅支持英文，格式为 Designer,Engineer",
        "日本語": "現在は英語のみで、Designer,Engineerのようにフォーマットされています",
        "한국어": "현재 영어만 사용 가능하며, Designer,Engineer와 같이 형식화됩니다"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    participants = st.text_area(placeholder=text,label="More Participants", value="",label_visibility="collapsed").replace("，", ",").split(",")
    st.session_state.more_participants = [] if participants == [''] else participants

with col1:

    language_map = {
        "English": "Topic",
        "中文": "主题",
        "日本語": "トピック",
        "한국어": "주제"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    st.subheader(text)
    language_map = {
        "English": "Enter a topic",
        "中文": "输入一个话题",
        "日本語": "トピックを入力",
        "한국어": "주제를 입력하세요"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    topic = st.text_input(text)
    language_map = {
        "English": "Discuss Settings",
        "中文": "讨论设置",
        "日本語": "ディスカッション設定",
        "한국어": "토론 설정"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    st.subheader(text)
    language_map = {
        "English": "Select participants (multiple options allowed)",
        "中文": "选择参与的人（允许多个选项）",
        "日本語": "参加者を選択（複数のオプションが許可されます）",
        "한국어": "참여자 선택(여러 옵션 허용)"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    chosen_people= st.multiselect(label=text,
        options= st.session_state.more_participants +
        ['Moderator','Artist', 'Mathematician', 'Designer', 'Engineer', 
                                      'Scientist','Writer','Philosopher','Historian',
                                      'Musician','Entrepreneur','Educator','Programmer',
                                      'Psychologist','Biologist','Chef','Athlete','Doctor',
                                      'Nurse','Lawyer','Politician','Journalist','Police','Soldier',
                                      'Firefighter','Farmer','Pilot','Driver','Singer','Dancer',
                                      'Actor','Model','Photographer','Athlete','Coach','Trainer',
                                      'Therapist','Counselor','Consultant','Advisor','Analyst',
                                      'Technician','Specialist','Expert','Assistant','Secretary',
                                      'Receptionist','Manager','Supervisor','Director','Leader','President'],default=["Moderator"])
    language_map = {
        "English": "Talk order",
        "中文": "谈话顺序",
        "日本語": "話し順",
        "한국어": "대화 순서"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    talk_order = st.selectbox(label=text,options=["Order","Random","Auto"],index=0)

    c1,c2 = st.columns([1,1])
    with c1:
        language_map = {
            "English": "Start Discussion",
            "中文": "开始讨论",
            "日本語": "話し合いを始める",
            "한국어": "토론 시작"
        }
        text = language_map.get(st.session_state.language, language_map["English"])
        if st.button(text):
            if not st.session_state.api_key and not st.session_state.base_url:
                st.toast("🚨 Please enter your API Key and Base URL in the sidebar")
            elif not topic:
                st.toast("🚨 Please enter a topic")
            elif not chosen_people:
                st.toast("🚨 Please choose who to discuss with")
            elif chosen_people == ["Moderator"]:
                st.toast("🚨 Please choose at least one more person to discuss with")
            else:
                st.session_state.start_discussion = True
                st.toast("🎉 Discussion started.")
                st.session_state.messages = [{"role": "user", "content": topic, "sender": "user"}]
                st.session_state.thread_id = Group._generate_thread_id()
                st.session_state.participants = [AgentSchema(name=person,
                                            transfer_to_me_description=f"I am a {person}, call me if you have any questions related to {person}.",
                                            agent=Agent(name=person,description=f"You are a {person},always reply in language {st.session_state.language}",
                                                        api_key=st.session_state.api_key,
                                                        base_url=st.session_state.base_url,
                                                        model=st.session_state.model
                                                        ),
                                            as_entry=True if person == "Moderator" else False) 
                                            for person in chosen_people]
                st.session_state.group = Group(participants=st.session_state.participants
                                               ,api_key=st.session_state.api_key
                                               ,base_url=st.session_state.base_url,
                                                  model=st.session_state.model)
    with c2:
        language_map = {
            "English": "Stop Discussion",
            "中文": "结束讨论",
            "日本語": "討論を終える",
            "한국어": "토론 정리"
        }
        text = language_map.get(st.session_state.language, language_map["English"])
        if st.button(text):
            st.toast("🎉 Discussion stopped.")
            st.session_state.messages = []
            st.session_state.start_discussion = False
            st.session_state.init_discussion = True
            st.rerun()
    language_map = {
        "English": "You can speak at any time",
        "中文": "你可以随时发言",
        "日本語": "いつでも発言可",
        "한국어": "언제든지 말할 수 있습니다"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    st.subheader(text)
    language_map = {
        "English": "Type your message here",
        "中文": "输入你的消息",
        "日本語": "メッセージを入力",
        "한국어": "메시지를 입력하세요"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    prompt = st.chat_input(text)

with col2:
    st.subheader("Discussion" if st.session_state.language == "English" else "讨论" if st.session_state.language == "中文" else "ディスカッション" if st.session_state.language == "日本語" else "토론")
    if chosen_people:
        language_map = {
            "English": "There are **{}** in this discussion. [ Select next person by **{}** ]",
            "中文": "这次讨论中有 **{}** 。[ 通过 **{}** 选择下一个人 ]",
            "日本語": "このディスカッションには **{}** がいます。[ **{}** で次の人を選択 ]",
            "한국어": "이 토론에는 **{}** 가 있습니다. [ **{}** 로 다음 사람 선택 ]"
            }
        caption_text = language_map.get(st.session_state.language, language_map["English"]).format(",".join(chosen_people), talk_order.lower())
        st.caption(caption_text)
    if st.session_state.start_discussion and st.session_state.group:
        next_agent = st.session_state.group.current_agent.get(st.session_state.thread_id,st.session_state.group.entry_agent).name
        if not st.session_state.skip_me and prompt:
            st.session_state.messages.append({"role": "user", "content":prompt, "sender": "user"})
            
            async def get_next_agent_auto():
                next_agent = await st.session_state.group.handoff(
                    messages=[message for message in st.session_state.messages if message["sender"] != "helper"][-3:],
                                                model=st.session_state.model,
                                                handoff_max_turns=1,
                                                include_current = False,
                                                next_speaker_select_mode="auto",
                                                thread_id=st.session_state.thread_id)
                language_map = {
                    "English": "Transfer to {}",
                    "中文": "转接给{}",
                    "日本語": "{} に転送",
                    "한국어": "{} 로 전환"
                }
                text = language_map.get(st.session_state.language, language_map["English"]).format(next_agent)
                st.session_state.messages.append({"role": "assistant", "content":text.format(next_agent), "sender": "helper"})
                message = build_message(st.session_state.messages,next_agent,topic,chosen_people)
                response = await st.session_state.group.current_agent.get(st.session_state.thread_id).agent.chat_async(message)
                st.session_state.messages.extend(response)
            with st.spinner('Discussion in progress...' if st.session_state.language == "English" else "讨论进行中..." if st.session_state.language == "中文" else "ディスカッション中..." if st.session_state.language == "日本語" else "토론 진행 중..."):
                asyncio.run(get_next_agent_auto())
                st.session_state.init_discussion = False

        else:
            st.session_state.skip_me = False
            async def get_next_agent(next_speaker_select_mode):
                if not st.session_state.init_discussion:
                    next_agent = await st.session_state.group.handoff(
                        messages=[message for message in st.session_state.messages if message["sender"] != "helper"][-3:],
                                                    model=st.session_state.model,
                                                    handoff_max_turns=1,
                                                    include_current = False,
                                                    next_speaker_select_mode=next_speaker_select_mode.lower(),
                                                    thread_id=st.session_state.thread_id)
                else:
                    next_agent = st.session_state.group.entry_agent.name
                language_map = {
                    "English": "Transfer to {}",
                    "中文": "转接给{}",
                    "日本語": "{} に転送",
                    "한국어": "{} 로 전환"
                }
                text = language_map.get(st.session_state.language, language_map["English"]).format(next_agent)
                st.session_state.messages.append({"role": "assistant", "content":text.format(next_agent), "sender": "helper"})
                message = build_message(st.session_state.messages,next_agent,topic,chosen_people)
                response = await st.session_state.group.current_agent.get(st.session_state.thread_id,st.session_state.group.entry_agent).agent.chat_async(message)
                st.session_state.messages.extend(response)
            with st.spinner('Discussion in progress...' if st.session_state.language == "English" else "讨论进行中..." if st.session_state.language == "中文" else "ディスカッション中..." if st.session_state.language == "日本語" else "토론 진행 중..."):
                asyncio.run(get_next_agent(talk_order))
                st.session_state.init_discussion = False

    else:
        if not st.session_state.api_key and not st.session_state.base_url:
            language_map = {
                "English": "Please enter your API Key and Base URL in the sidebar.",
                "中文": "请在侧边栏中输入您的API Key和Base URL。",
                "日本語": "サイドバーにAPIキーとベースURLを入力してください。",
                "한국어": "사이드바에 API 키와 기본 URL을 입력하세요."
            }
            text = language_map.get(st.session_state.language, language_map["English"])
            st.warning(text)
        if not topic:
            language_map = {
                "English": "Please enter a topic.",
                "中文": "请输入一个话题。",
                "日本語": "トピックを入力してください。",
                "한국어": "주제를 입력하세요."
            }
            text = language_map.get(st.session_state.language, language_map["English"])
            st.warning(text)
        if not chosen_people:
            language_map = {
                "English": "Please choose who to discuss with.",
                "中文": "请选择要讨论的人。",
                "日本語": "話し合う相手を選択してください。",
                "한국어": "누구와 토론할지 선택하세요."
            }
            text = language_map.get(st.session_state.language, language_map["English"])
            st.warning(text)
        if chosen_people == ["Moderator"]:
            language_map = {
                "English": "Please choose at least one more person to discuss with.",
                "中文": "请至少再选择一个人进行讨论。",
                "日本語": "少なくとももう一人を選んで議論してください。",
                "한국어": "최소한 한 사람을 더 선택하여 토론해 주세요."
            }
            text = language_map.get(st.session_state.language, language_map["English"])
            st.warning(text)

        language_map = {
            "English": "Press the **Start Discussion** button to start a discussion.",
            "中文": "按下 **开始讨论** 按钮开始讨论。",
            "日本語": "**話し合いを始める** ボタンを押して話し合いを始めてください。",
            "한국어": "**토론 시작** 버튼을 눌러 토론을 시작하세요."
        }
        text = language_map.get(st.session_state.language, language_map["English"])
        st.info(text)
    
    for index,message in enumerate(st.session_state.messages):
        if "sender" in message and message["sender"] == "helper":
            with st.chat_message("ai"):
                st.write(message["content"].split(",")[0])
        else:
            with st.chat_message(message["sender"] if "sender" in message else message["role"]):
                st.write(message["content"])
                if index == len(st.session_state.messages) - 1:
                    language_map = {
                        "English": "Next Person",
                        "中文": "下一个人",
                        "日本語": "次の人",
                        "한국어": "다음 사람"
                    }
                    text = language_map.get(st.session_state.language, language_map["English"])
                    st.button(label=text,on_click=skip_me, key="next_person")