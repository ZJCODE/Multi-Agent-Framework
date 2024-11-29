import streamlit as st
from agent import Group, AgentSchema,Agent
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

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

if "more_participants_translate" not in st.session_state:
    st.session_state.more_participants_translate = []

if "language" not in st.session_state:
    st.session_state.language = "English"

def skip_me():
    st.session_state.skip_me = True

def restart_discussion():
    st.session_state.messages = []
    st.session_state.start_discussion = False
    st.session_state.init_discussion = True
    st.session_state.more_participants = []
    st.session_state.more_participants_translate = []

@st.cache_data
def translate2english(text,api_key,base_url,model):
        client = OpenAI(api_key=api_key, base_url=base_url)
        response =  client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": """
                       Translate text to English

                       ## Example 1
                        - Input: 编剧
                        - Output: Screenwriter
                       ## Example 2
                        - Input: writer
                        - Output: Writer 
                       ## Example 3
                        - Input: 导演，演员
                        - Output: Director,Actor
                       ## Example 4
                        - Input: 设计师，工程师，医生
                        - Output: Designer,Engineer,Doctor

                       ## Task
                        - Translate the following text to English
                       """
                       }] + [{"role": "user", "content": text}],
            stream=False
            )
        return response.choices[0].message.content


def build_message(messages_history, current_speaker,topic,participants=[],max_message_length=10):
    current_speaker_message = [message for message in messages_history if message["sender"] == current_speaker]
    other_people_messages = [message for message in messages_history if message["sender"] not in ["helper",current_speaker]]

    current_speaker_message = current_speaker_message[-max_message_length:]
    other_people_messages = other_people_messages[-max_message_length:]
    
    prompt = """# Discussion on Topic: {}

### Participants

{}

### Your Previous Opinions

{}

### Other people's Opinions

{}

### Task

Consider the previous opinions in the discussion. As a {}, it's your turn to speak.""".format(topic,
                ",".join(participants),
                "\n\n".join([f"{message['content']}" for message in current_speaker_message]),
                "\n\n".join([f"{message['content']}" for message in other_people_messages]),
                current_speaker)
    
    if current_speaker == 'Moderator':
        prompt += "\n\nAs a moderator, you can ask questions, summarize the discussion, or guide the conversation. if there is no previous message, you can start the discussion."
    
    return prompt

def build_handoff_message(messages_history,topic,participants=[]):
    spoken_history_counter = dict(zip(participants,[0]*len(participants)))
    moderator_messages = [message for message in messages_history if message["sender"] == "Moderator"]
    messages_filtered = [message for message in messages_history if message["sender"] not in ["helper","Moderator"]]
    for message in messages_history:
        if message["sender"] != "helper":
            spoken_history_counter[message["sender"]] += 1
    prompt = """### People's Spoken History

{}

### Recent Message

{}

{}

### Task

Utilize the latest message and the individuals who have already participated in the conversation to determine the most appropriate person to speak next.""".format(
                json.dumps(spoken_history_counter,indent=4),
                moderator_messages[-1]['sender'] +  " said:\n\n " + moderator_messages[-1]['content'] if moderator_messages else "",
                "\n\n".join([f"\n{message['sender']} said:\n\n {message['content']}\n" for message in messages_filtered[-2:]]) if messages_filtered else "")
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
        "English": "Formatted as Designer,Engineer",
        "中文": "格式为 设计师,工程师",
        "日本語": "デザイナー、エンジニアなどの形式",
        "한국어": "디자이너,엔지니어 형식"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    participants_raw = st.text_area(placeholder=text,label="More Participants", 
                                    value=",".join(st.session_state.more_participants),
                                    label_visibility="collapsed")
    language_map = {
        "English": "Add",
        "中文": "添加",
        "日本語": "追加",
        "한국어": "추가"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    if st.button(text):
        if participants_raw:
            participants = participants_raw.replace("，", ",").split(",")

            with st.spinner('Adding participants...' if st.session_state.language == "English" else "添加参与者中..." if st.session_state.language == "中文" else "参加者を追加中..." if st.session_state.language == "日本語" else "참가자 추가 중..."):
                participants_translate = translate2english(participants_raw,st.session_state.api_key,st.session_state.base_url,st.session_state.model).replace("，", ",").split(",")
                if len(participants) != len(participants_translate):
                    language_map = {
                        "English": "Please formatted as Designer,Engineer",
                        "中文": "请按照 设计师,工程师 的格式 输入",
                        "日本語": "デザイナー、エンジニアなどの形式で入力してください",
                        "한국어": "디자이너,엔지니어 형식으로 입력해주세요"
                    }
                    text = language_map.get(st.session_state.language, language_map["English"])
                    st.warning(text)
                else:
                    st.session_state.more_participants = [] if participants == [''] else participants
                    st.session_state.more_participants_translate = [] if participants_translate == [''] else [x.strip() for x in  participants_translate]
                    st.warning(st.session_state.more_participants_translate)
                    st.warning(st.session_state.more_participants)
        else:
            st.session_state.more_participants = []
            st.session_state.more_participants_translate = []
            st.warning("Please input participants")

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
    participants_options_map = {
        "English": st.session_state.more_participants + ["Moderator","Mathematician","Artist","Historian","Scientist","Writer","Poet","Musician","Philosopher","Sociologist","Psychologist","Educator","Linguist","Anthropologist","Political Scientist","Economist","Environmentalist","Designer","Engineer","Doctor","Nurse","Architect","Programmer","Data Analyst","Nutritionist","Psychotherapist","Pharmacist","Physical Therapist","Environmental Engineer","Urban Planner","Mechanical Engineer","Electrical Engineer","Executive","Technical Expert","Marketing Specialist","Financial Analyst","Human Resources Manager","Legal Advisor","Public Relations Specialist","Customer Representative","Supply Chain Management Specialist","Researcher","Policy Maker","Entrepreneur","Investor","Financial Advisor","Corporate Social Responsibility Specialist"],
        "中文": st.session_state.more_participants + ["主持人","数学家","艺术家","历史学家","科学家","作家","诗人","音乐家","哲学家","社会学家","心理学家","教育家","语言学家","人类学家","政治学家","经济学家","环境学家","设计师","工程师","医生","护士","建筑师","程序员","数据分析师","营养师","心理治疗师","药剂师","物理治疗师","环境工程师","城市规划师","机械工程师","电气工程师","企业高管","技术专家","市场营销专家","财务分析师","人力资源经理","法律顾问","公共关系专家","客户代表","供应链管理专家","研究员","政策制定者","创业者","投资者","金融顾问","社会责任专家"],
        "日本語": st.session_state.more_participants + ["モデレーター","数学者","アーティスト","歴史家","科学者","作家","詩人","音楽家","哲学者","社会学者","心理学者","教育者","言語学者","人類学者","政治学者","経済学者","環境保護活動家","デザイナー","エンジニア","医者","看護師","建築家","プログラマー","データアナリスト","栄養士","心理療法士","薬剤師","理学療法士","環境エンジニア","都市計画家","機械工学者","電気工学者","役員","技術専門家","マーケティングスペシャリスト","財務アナリスト","人事マネージャー","法律顧問","広報スペシャリスト","カスタマー代表","サプライチェーン管理スペシャリスト","研究者","政策立案者","起業家","投資家","ファイナンシャルアドバイザー","企業の社会的責任スペシャリスト"],
        "한국어": st.session_state.more_participants + ["모더레이터","수학자","예술가","역사학자","과학자","작가","시인","음악가","철학자","사회학자","심리학자","교육자","언어학자","인류학자","정치학자","경제학자","환경운동가","디자이너","엔지니어","의사","간호사","건축가","프로그래머","데이터 분석가","영양사","심리치료사","약사","물리치료사","환경 엔지니어","도시 계획가","기계공학자","전기공학자","임원","기술 전문가","마케팅 전문가","재무 분석가","인사 관리자","법률 고문","홍보 전문가","고객 대표","공급망 관리 전문가","연구원","정책 입안자","기업가","투자자","재무 고문","기업의 사회적 책임 전문가"]
    }

    default_participant_map = {
        "English": ["Moderator"],
        "中文": ["主持人"],
        "日本語": ["モデレーター"],
        "한국어": ["모더레이터"]
    }
    participants_language_map = {
        "Moderator":"Moderator","Mathematician":"Mathematician","Artist":"Artist","Historian":"Historian","Scientist":"Scientist","Writer":"Writer","Poet":"Poet","Musician":"Musician","Philosopher":"Philosopher","Sociologist":"Sociologist","Psychologist":"Psychologist","Educator":"Educator","Linguist":"Linguist","Anthropologist":"Anthropologist","Political Scientist":"Political Scientist","Economist":"Economist","Environmentalist":"Environmentalist","Designer":"Designer","Engineer":"Engineer","Doctor":"Doctor","Nurse":"Nurse","Architect":"Architect","Programmer":"Programmer","Data Analyst":"Data Analyst","Nutritionist":"Nutritionist","Psychotherapist":"Psychotherapist","Pharmacist":"Pharmacist","Physical Therapist":"Physical Therapist","Environmental Engineer":"Environmental Engineer","Urban Planner":"Urban Planner","Mechanical Engineer":"Mechanical Engineer","Electrical Engineer":"Electrical Engineer","Executive":"Executive","Technical Expert":"Technical Expert","Marketing Specialist":"Marketing Specialist","Financial Analyst":"Financial Analyst","Human Resources Manager":"Human Resources Manager","Legal Advisor":"Legal Advisor","Public Relations Specialist":"Public Relations Specialist","Customer Representative":"Customer Representative","Supply Chain Management Specialist":"Supply Chain Management Specialist","Researcher":"Researcher","Policy Maker":"Policy Maker","Entrepreneur":"Entrepreneur","Investor":"Investor","Financial Advisor":"Financial Advisor","Corporate Social Responsibility Specialist":"Corporate Social Responsibility Specialist",
        "主持人":"Moderator","数学家":"Mathematician","艺术家":"Artist","历史学家":"Historian","科学家":"Scientist","作家":"Writer","诗人":"Poet","音乐家":"Musician","哲学家":"Philosopher","社会学家":"Sociologist","心理学家":"Psychologist","教育家":"Educator","语言学家":"Linguist","人类学家":"Anthropologist","政治学家":"Political Scientist","经济学家":"Economist","环境保护活动家":"Environmentalist","设计师":"Designer","工程师":"Engineer","医生":"Doctor","护士":"Nurse","建筑师":"Architect","程序员":"Programmer","数据分析师":"Data Analyst","营养师":"Nutritionist","心理治疗师":"Psychotherapist","药剂师":"Pharmacist","理学疗法师":"Physical Therapist","环境工程师":"Environmental Engineer","城市规划师":"Urban Planner","机械工程师":"Mechanical Engineer","电气工程师":"Electrical Engineer","执行":"Executive","技术专家":"Technical Expert","市场营销专家":"Marketing Specialist","财务分析师":"Financial Analyst","人力资源经理":"Human Resources Manager","法律顾问":"Legal Advisor","公共关系专家":"Public Relations Specialist","客户代表":"Customer Representative","供应链管理专家":"Supply Chain Management Specialist","研究员":"Researcher","政策制定者":"Policy Maker","创业者":"Entrepreneur","投资者":"Investor","金融顾问":"Financial Advisor","企业社会责任专家":"Corporate Social Responsibility Specialist",
        "モデレーター":"Moderator","数学者":"Mathematician","アーティスト":"Artist","歴史家":"Historian","科学者":"Scientist","作家":"Writer","詩人":"Poet","音楽家":"Musician","哲学者":"Philosopher","社会学者":"Sociologist","心理学者":"Psychologist","教育者":"Educator","言語学者":"Linguist","人類学者":"Anthropologist","政治学者":"Political Scientist","経済学者":"Economist","環境保護活動家":"Environmentalist","デザイナー":"Designer","エンジニア":"Engineer","医者":"Doctor","看護師":"Nurse","建築家":"Architect","プログラマー":"Programmer","データアナリスト":"Data Analyst","栄養士":"Nutritionist","心理療法士":"Psychotherapist","薬剤師":"Pharmacist","理学療法士":"Physical Therapist","環境エンジニア":"Environmental Engineer","都市計画家":"Urban Planner","機械工学者":"Mechanical Engineer","電気工学者":"Electrical Engineer","役員":"Executive","技術専門家":"Technical Expert","マーケティングスペシャリスト":"Marketing Specialist","財務アナリスト":"Financial Analyst","人事マネージャー":"Human Resources Manager","法律顧問":"Legal Advisor","広報スペシャリスト":"Public Relations Specialist","カスタマー代表":"Customer Representative","サプライチェーン管理スペシャリスト":"Supply Chain Management Specialist","研究者":"Researcher","政策立案者":"Policy Maker","起業家":"Entrepreneur","投資家":"Investor","ファイナンシャルアドバイザー":"Financial Advisor","企業の社会的責任スペシャリスト":"Corporate Social Responsibility Specialist",
        "모더레이터":"Moderator","수학자":"Mathematician","예술가":"Artist","역사학자":"Historian","과학자":"Scientist","작가":"Writer","시인":"Poet","음악가":"Musician","철학자":"Philosopher","사회학자":"Sociologist","심리학자":"Psychologist","교육자":"Educator","언어학자":"Linguist","인류학자":"Anthropologist","정치학자":"Political Scientist","경제학자":"Economist","환경운동가":"Environmentalist","디자이너":"Designer","엔지니어":"Engineer","의사":"Doctor","간호사":"Nurse","건축가":"Architect","프로그래머":"Programmer","데이터 분석가":"Data Analyst","영양사":"Nutritionist","심리치료사":"Psychotherapist","약사":"Pharmacist","물리치료사":"Physical Therapist","환경 엔지니어":"Environmental Engineer","도시 계획가":"Urban Planner","기계공학자":"Mechanical Engineer","전기공학자":"Electrical Engineer","임원":"Executive","기술 전문가":"Technical Expert","마케팅 전문가":"Marketing Specialist","재무 분석가":"Financial Analyst","인사 관리자":"Human Resources Manager","법률 고문":"Legal Advisor","홍보 전문가":"Public Relations Specialist","고객 대표":"Customer Representative","공급망 관리 전문가":"Supply Chain Management Specialist","연구원":"Researcher","정책 입안자":"Policy Maker","기업가":"Entrepreneur","투자자":"Investor","재무 고문":"Financial Advisor","기업의 사회적 책임 전문가":"Corporate Social Responsibility Specialist"}


    for i in range(len(st.session_state.more_participants)):
        participants_language_map[st.session_state.more_participants[i]] = st.session_state.more_participants_translate[i]

    participants_language_reverse_map = {
        "Moderator":{"English":"Moderator","中文":"主持人","日本語":"モデレーター","한국어":"모더레이터"},
        "Mathematician":{"English":"Mathematician","中文":"数学家","日本語":"数学者","한국어":"수학자"},
        "Artist":{"English":"Artist","中文":"艺术家","日本語":"アーティスト","한국어":"예술가"},
        "Historian":{"English":"Historian","中文":"历史学家","日本語":"歴史家","한국어":"역사학자"},
        "Scientist":{"English":"Scientist","中文":"科学家","日本語":"科学者","한국어":"과학자"},
        "Writer":{"English":"Writer","中文":"作家","日本語":"作家","한국어":"작가"},
        "Poet":{"English":"Poet","中文":"诗人","日本語":"詩人","한국어":"시인"},
        "Musician":{"English":"Musician","中文":"音乐家","日本語":"音楽家","한국어":"음악가"},
        "Philosopher":{"English":"Philosopher","中文":"哲学家","日本語":"哲学者","한국어":"철학자"},
        "Sociologist":{"English":"Sociologist","中文":"社会学家","日本語":"社会学者","한국어":"사회학자"},
        "Psychologist":{"English":"Psychologist","中文":"心理学家","日本語":"心理学者","한국어":"심리학자"},
        "Educator":{"English":"Educator","中文":"教育家","日本語":"教育者","한국어":"교육자"},
        "Linguist":{"English":"Linguist","中文":"语言学家","日本語":"言語学者","한국어":"언어학자"},
        "Anthropologist":{"English":"Anthropologist","中文":"人类学家","日本語":"人類学者","한국어":"인류학자"},
        "Political Scientist":{"English":"Political Scientist","中文":"政治学家","日本語":"政治学者","한국어":"정치학자"},
        "Economist":{"English":"Economist","中文":"经济学家","日本語":"経済学者","한국어":"경제학자"},
        "Environmentalist":{"English":"Environmentalist","中文":"环境保护活动家","日本語":"環境保護活動家","한국어":"환경운동가"},
        "Designer":{"English":"Designer","中文":"设计师","日本語":"デザイナー","한국어":"디자이너"},
        "Engineer":{"English":"Engineer","中文":"工程师","日本語":"エンジニア","한국어":"엔지니어"},
        "Doctor":{"English":"Doctor","中文":"医生","日本語":"医者","한국어":"의사"},
        "Nurse":{"English":"Nurse","中文":"护士","日本語":"看護師","한국어":"간호사"},
        "Architect":{"English":"Architect","中文":"建筑师","日本語":"建築家","한국어":"건축가"},
        "Programmer":{"English":"Programmer","中文":"程序员","日本語":"プログラマー","한국어":"프로그래머"},
        "Data Analyst":{"English":"Data Analyst","中文":"数据分析师","日本語":"データアナリスト","한국어":"데이터 분석가"},
        "Nutritionist":{"English":"Nutritionist","中文":"营养师","日本語":"栄養士","한국어":"영양사"},
        "Psychotherapist":{"English":"Psychotherapist","中文":"心理治疗师","日本語":"心理療法士","한국어":"심리치료사"},
        "Pharmacist":{"English":"Pharmacist","中文":"药剂师","日本語":"薬剤師","한국어":"약사"},
        "Physical Therapist":{"English":"Physical Therapist","中文":"理学疗法师","日本語":"理学療法士","한국어":"물리치료사"},
        "Environmental Engineer":{"English":"Environmental Engineer","中文":"环境工程师","日本語":"環境エンジニア","한국어":"환경 엔지니어"},
        "Urban Planner":{"English":"Urban Planner","中文":"城市规划师","日本語":"都市計画家","한국어":"도시 계획가"},
        "Mechanical Engineer":{"English":"Mechanical Engineer","中文":"机械工程师","日本語":"機械工学者","한국어":"기계공학자"},
        "Electrical Engineer":{"English":"Electrical Engineer","中文":"电气工程师","日本語":"電気工学者","한국어":"전기공학자"},
        "Executive":{"English":"Executive","中文":"执行","日本語":"役員","한국어":"임원"},
        "Technical Expert":{"English":"Technical Expert","中文":"技术专家","日本語":"技術専門家","한국어":"기술 전문가"},
        "Marketing Specialist":{"English":"Marketing Specialist","中文":"市场营销专家","日本語":"マーケティングスペシャリスト","한국어":"마케팅 전문가"},
        "Financial Analyst":{"English":"Financial Analyst","中文":"财务分析师","日本語":"財務アナリスト","한국어":"재무 분석가"},
        "Human Resources Manager":{"English":"Human Resources Manager","中文":"人力资源经理","日本語":"人事マネージャー","한국어":"인사 관리자"},
        "Legal Advisor":{"English":"Legal Advisor","中文":"法律顾问","日本語":"法律顧問","한국어":"법률 고문"},
        "Public Relations Specialist":{"English":"Public Relations Specialist","中文":"公共关系专家","日本語":"広報スペシャリスト","한국어":"홍보 전문가"},
        "Customer Representative":{"English":"Customer Representative","中文":"客户代表","日本語":"カスタマー代表","한국어":"고객 대표"},
        "Supply Chain Management Specialist":{"English":"Supply Chain Management Specialist","中文":"供应链管理专家","日本語":"サプライチェーン管理スペシャリスト","한국어":"공급망 관리 전문가"},
        "Researcher":{"English":"Researcher","中文":"研究员","日本語":"研究者","한국어":"연구원"},
        "Policy Maker":{"English":"Policy Maker","中文":"政策制定者","日本語":"政策立案者","한국어":"정책 입안자"},
        "Entrepreneur":{"English":"Entrepreneur","中文":"创业者","日本語":"起業家","한국어":"기업가"},
        "Investor":{"English":"Investor","中文":"投资者","日本語":"投資家","한국어":"투자자"},
        "Financial Advisor":{"English":"Financial Advisor","中文":"金融顾问","日本語":"ファイナンシャルアドバイザー","한국어":"재무 고문"},
        "Corporate Social Responsibility Specialist":{"English":"Corporate Social Responsibility Specialist","中文":"企业社会责任专家","日本語":"企業の社会的責任スペシャリスト","한국어":"기업의 사회적 책임 전문가"}
    }

        # st.session_state.more_participants_translate
        # st.seeeion_state.more_participants
        # update participants_language_reverse_map
    for i in range(len(st.session_state.more_participants)):
        if st.session_state.more_participants_translate[i] not in participants_language_reverse_map:
            participants_language_reverse_map[st.session_state.more_participants_translate[i]] = {st.session_state.language:st.session_state.more_participants[i]}
        else:
            participants_language_reverse_map[st.session_state.more_participants_translate[i]][st.session_state.language] = st.session_state.more_participants[i]

    options = participants_options_map.get(st.session_state.language, participants_options_map["English"])
    default_participant = default_participant_map.get(st.session_state.language, default_participant_map["English"])
    chosen_people_original= st.multiselect(label=text,options= options,default=default_participant)

    chosen_people = [participants_language_map.get(person,person) for person in chosen_people_original]

    language_map = {
        "English": "Talk order",
        "中文": "谈话顺序",
        "日本語": "話し順",
        "한국어": "대화 순서"
    }
    text = language_map.get(st.session_state.language, language_map["English"])
    order_language_map = {
        "English": ["Order","Random","Auto"],
        "中文": ["顺序","随机","自动"],
        "日本語": ["順序","ランダム","自動"],
        "한국어": ["순서","랜덤","자동"]
    }
    order_text = order_language_map.get(st.session_state.language, order_language_map["English"])
    talk_order_original = st.selectbox(label=text,options=order_text,index=0)

    talk_order_map = {
        "Order": "Order","Random": "Random","Auto": "Auto",
        "顺序": "Order","随机": "Random","自动": "Auto",
        "順序": "Order","ランダム": "Random","自動": "Auto",
        "순서": "Order","랜덤": "Random","자동": "Auto"
    }
    talk_order = talk_order_map.get(talk_order_original, "Order")


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
                # st.session_state.messages = [{"role": "user", "content": topic, "sender": "user"}]
                st.session_state.thread_id = Group._generate_thread_id()
                st.session_state.participants = [AgentSchema(name=person.replace(" ","_"),
                                            transfer_to_me_description=f"I am a {person}, call me if you have any questions related to {person}.",
                                            agent=Agent(name=person.replace(" ","_"),description=f"You are a {person},reply in short form and always reply in language {st.session_state.language}",
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
    if chosen_people_original:
        language_map = {
            "English": "There are **{}** in this discussion. [ Select next person by **{}** ]",
            "中文": "这次讨论中有 **{}** 。[ 通过 **{}** 选择下一个人 ]",
            "日本語": "このディスカッションには **{}** がいます。[ **{}** で次の人を選択 ]",
            "한국어": "이 토론에는 **{}** 가 있습니다. [ **{}** 로 다음 사람 선택 ]"
            }
        caption_text = language_map.get(st.session_state.language, language_map["English"]).format(",".join(chosen_people_original), talk_order_original)
        st.caption(caption_text)


    for index,message in enumerate(st.session_state.messages):
        if "sender" in message and message["sender"] == "helper":
            with st.chat_message("ai"):
                st.markdown(message["content"].split(",")[0])
        else:
            with st.chat_message(message["sender"] if "sender" in message else message["role"]):
                st.markdown(message["content"])

    if st.session_state.start_discussion and st.session_state.group:
        next_agent = st.session_state.group.current_agent.get(st.session_state.thread_id,st.session_state.group.entry_agent).name
        if not st.session_state.skip_me and prompt:
            language_map = {
                "English": "User Participated",
                "中文": "用户参与",
                "日本語": "ユーザーが参加しました",
                "한국어": "사용자 참여"
            }
            text = language_map.get(st.session_state.language, language_map["English"])
            st.session_state.messages.append({"role": "assistant", "content":prompt, "sender": "helper"})
            st.session_state.messages.append({"role": "user", "content":prompt, "sender": "user"})
            with st.chat_message("ai"):
                st.markdown(text)
            with st.chat_message("user"):
                st.markdown(prompt)
            st.warning(build_handoff_message(st.session_state.messages,topic,chosen_people))
            next_agent = st.session_state.group.handoff(
                messages=build_handoff_message(st.session_state.messages,topic,chosen_people),
                                            model=st.session_state.model,
                                            handoff_max_turns=1,
                                            include_current = False,
                                            next_speaker_select_mode="auto",
                                            thread_id=st.session_state.thread_id)
            language_map = {
                "English": "Transfer to {}",
                "中文": "转接给 {}",
                "日本語": "{} に転送",
                "한국어": "{} 로 전환"
            }
            text = language_map.get(st.session_state.language, language_map["English"]).format(participants_language_reverse_map.get(next_agent.replace("_"," ")).get(st.session_state.language, next_agent.replace("_"," ")))
            st.session_state.messages.append({"role": "assistant", "content":text, "sender": "helper"})
            with st.chat_message("ai"):
                st.markdown(text)
            message = build_message(st.session_state.messages,next_agent,topic,chosen_people_original)
            stream = st.session_state.group.current_agent.get(st.session_state.thread_id).agent.chat(message,stream=True)
            with st.chat_message(next_agent):
                response = st.write_stream(stream)
                language_map = {
                    "English": "Next Person",
                    "中文": "下一个人",
                    "日本語": "次の人",
                    "한국어": "다음 사람"
                }
                text = language_map.get(st.session_state.language, language_map["English"])
                st.button(label=text,on_click=skip_me, key="next_person")
            st.session_state.messages.append({"role": "assistant", "content":response, "sender": next_agent})
            st.session_state.init_discussion = False
        else:
            st.session_state.skip_me = False
            if not st.session_state.init_discussion:
                st.warning(build_handoff_message(st.session_state.messages,topic,chosen_people))
                next_agent = st.session_state.group.handoff(
                    messages=build_handoff_message(st.session_state.messages,topic,chosen_people),
                                                model=st.session_state.model,
                                                handoff_max_turns=1,
                                                include_current = False,
                                                next_speaker_select_mode=talk_order.lower(),
                                                thread_id=st.session_state.thread_id)
            else:
                next_agent = st.session_state.group.entry_agent.name
            language_map = {
                "English": "Transfer to {}",
                "中文": "转接给 {}",
                "日本語": "{} に転送",
                "한국어": "{} 로 전환"
            }
            text = language_map.get(st.session_state.language, language_map["English"]).format(participants_language_reverse_map.get(next_agent.replace("_"," ")).get(st.session_state.language, next_agent.replace("_"," ")))
            st.session_state.messages.append({"role": "assistant", "content":text, "sender": "helper"})
            with st.chat_message("ai"):
                st.markdown(text)
            message = build_message(st.session_state.messages,next_agent,topic,chosen_people_original)
            stream = st.session_state.group.current_agent.get(st.session_state.thread_id,st.session_state.group.entry_agent).agent.chat(message,stream=True)
            with st.chat_message(next_agent):
                response = st.write_stream(stream)
                language_map = {
                    "English": "Next Person",
                    "中文": "下一个人",
                    "日本語": "次の人",
                    "한국어": "다음 사람"
                }
                text = language_map.get(st.session_state.language, language_map["English"])
                st.button(label=text,on_click=skip_me, key="next_person_")
            st.session_state.messages.append({"role": "assistant", "content":response, "sender": next_agent})
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
    