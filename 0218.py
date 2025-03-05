from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages.chat import ChatMessage
from module.tools_0218 import (
    product_recommend,
    site_search,
    Retriever,
    negotiation,
)
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_teddynote import logging
import streamlit as st
import redis
import warnings
import os

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# env 파일에서 OPENAI API KEY 들여옴
load_dotenv()

# LangChain 추적 시작
logging.langsmith("0217")

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

st.set_page_config(page_title="Buyer Agent", page_icon="🍽️", layout="wide")
st.title("이제 쇼핑은 쉽고 간편하게 당신을 위한 쇼핑 대리인")
st.markdown(
    "안녕하세요! 😊 구매자를 위한 에이전트입니다. 사고 싶은 제품을 입력해주시면, 그 제품에 대한 정보와 판매점 정보를 정성껏 알려드릴게요. 🛍️ 또한, 구매자를 대신해 판매자와 협상하고 구매를 진행해드리니 걱정하지 마세요! 💪✨"
)

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# Chain 저장용
if "agent" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["agent"] = None

# 구매자 정보 저장용
if "user_information" not in st.session_state:
    st.session_state["user_information"] = []

# 사이드바 생성
with st.sidebar:
    st.header("옵션💡")
    # 초기화 버튼 생성
    clear_btn = st.button("대화 다시 시작")

    st.header("사용자 정보💡")
    selected_user = st.selectbox("사용자 유형", ["구매자", "판매자"], index=0)

    # 구매자 정보
    if selected_user == "구매자":
        user_name = st.text_input(
            label=f"{selected_user} 이름",
            value=st.session_state.get("user_name", ""),
            key="user_name",
        )
    else:
        user_name = st.text_input(
            label=f"{selected_user} 이름 (판매 사이트 이름 입력)",
            value=st.session_state.get("user_name", ""),
            key="user_name",
        )
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("정보 설정", key="add_domain"):
            if user_name not in st.session_state["user_information"]:
                st.session_state["user_information"].append({"user_name": user_name})

    # 현재 등록된 배송 정보 및 결제 정보 목록 표시
    st.write("사용자 정보 목록:")
    for idx, domain in enumerate(st.session_state["user_information"]):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text(f"{domain['user_name']}")
        with col2:
            if st.button("삭제", key=f"del_{idx}"):
                st.session_state["user_information"].pop(idx)
                st.rerun()

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_ids, url=REDIS_URL
    )  # 해당 세션 ID에 대한 세션 기록 반환


# Redis 연결
# Redis 클라이언트 생성
client = redis.Redis.from_url(REDIS_URL)
session_id = client.keys("*")  # 모든 세션 Key 가져오기
all_session_id = [session_id_buyer.decode("utf-8") for session_id_buyer in session_id]


user_nego_chat = []
specific_session_id = None

# user_name 문자열 정규화
normalized_user_name = user_name.replace(" ", "").lower()

if selected_user == "판매자":
    for i in all_session_id:
        if "buyer" in i:
            history = get_session_history(i[14:])
            messages = history.messages
            if not messages:
                continue

            elif any(
                normalized_user_name in message.content.replace(" ", "").lower()
                for message in messages
            ):
                specific_session_id = i

                for message in messages:
                    if isinstance(message, HumanMessage):
                        role = "human"
                    elif isinstance(message, AIMessage):
                        role = "ai"
                    else:
                        role = "system"
                    user_nego_chat.append(f"[{role}] {message.content}")

                break

user_chat = []
if selected_user == "구매자":
    chat_history = get_session_history(f"buyer_{user_name}")
    messages = chat_history.messages
    if not messages:
        print(f"No messages found for buyer {user_name}")
    else:
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "human"
            elif isinstance(message, AIMessage):
                role = "ai"
            else:
                role = "system"
            user_chat.append(f"[{role}] {message.content}")


# Agent parser 정의
agent_stream_parser = AgentStreamParser()


# agent 생성
def buy_user_create_agent():
    tools = [product_recommend, site_search, Retriever]
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    # prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a Buyer Agent. Your primary role is to assist the buyer by exploring products, negotiating with sellers, and completing purchases on behalf of the buyer. 
                Follow the rules and formats below for all interactions with the buyer and seller.

                # Rules:

                ## When interacting with the buyer:
                - Use a friendly and warm tone throughout the conversation to make them feel comfortable and valued.
                - Incorporate emojis 😊 to add a cheerful and engaging touch to your communication.
                - Maintain professionalism while ensuring the conversation feels personal and approachable.
                - {user_name}을 부르면서 대화를 이어가세요.

                ### 1. Product Information Exploration
                - When the buyer requests a specific product, follow these steps:
                    - Ask for Additional Details: Gather more specific information about the product to understand the buyer's preferences.
                        (예: "어떤 브랜드나 가격대를 선호하시나요?")
                    - Provide Recommendations: Based on the information provided, use the product_recommend tool to suggest suitable products that match their preferences.
                        - [the product_recommend tool`s result]를 기반으로 사용자가 원하는 제품을 5개 추천하세요.
                        - Print the product name as the product line name or series name. 
                            - (예: 나이키 머큐리얼 베이퍼 14 아카데미 AG(product`s Full Name) -> 나이키 머큐리얼 베이퍼(product`s line name or series name))  
                        - However, the output format is as follows
                        
                        # Answer Format:
                        1. 상품 이름 (Print the product name as the product line name or series name.)
                        - 상품 특징:
                        - 추천 이유:
                        - URL: (URL만 제시)
                        2. 상품 이름 (Print the product name as the product line name or series name.)
                        - 상품 특징:
                        - 추천 이유:
                        - URL: (URL만 제시)
                        3. 상품 이름 (Print the product name as the product line name or series name.)
                        - 상품 특징:
                        - 추천 이유:
                        - URL: (URL만 제시)
                        ....

                ### 2. Providing Online seller
                When a buyer selects a specific product from the recommendations, follow these steps:
                - Use site_search tool to identify retailers offering the selected product and present this information to the buyer:
                    - When providing retailer details, include the unique characteristics of each retailer.
                    - When using site_search to find retailers, set the query parameter to "[selected product`s category] 전문 온라인 판매점"
                        - 예를 들어, 구매자가 '나이키 머큐리얼 베이퍼'라는 축구화를 원한다면, '축구화 전문 온라인 판매점'으로 검색하세요.
                    - Answer Format is as follows
                    # Answer Format:
                    1. 판매처1
                    - 판매처1 특징:
                    - 판매처1 URL:
                    
                    2. 판매처2
                    - 판매처2 특징:
                    - 판매처2 URL:

                    3. 판매처3
                    - 판매처3 특징:
                    - 판매처3 URL:  
                    ....

                ### 3. Crawling Data from Specialized Shopping Sites (Use Retriever Tool)
                When a buyer selects a specific seller from the provided seller, follow these steps:
                - Pass all the full name of the shopping sites to the sites parameter.
                    - Ex, all the full name of the shopping sites are '크레이지 11', '레드사커' -> sites = ['크레이지 11', '레드사커']
                - Pass the selected product name to the product_name parameter.
                - Extract product-related data from the site to ensure comprehensive coverage of available options.
                - Query the Retriever:
                    - Use the Retriever tool to retrieve product information based on the crawled data.
                    - Set the query parameter of the Retriever tool to the query parameter of the product_recommend tool.
                    - This ensures that product details are accurately retrieved using the crawled data.
                    - 각각의 [판매처]에 대한 상품 데이터를 제공하세요.
                    - 검색을 했는데 사용자가 원하는 데이터가 없으면, '[판매처]에 관련 제품이 존재하지만, {user_name}님께서 원하는 가격 혹은 기능의 제품은 없습니다'라고 답변하세요.
                - Present the Information to the Buyer:
                    - Include essential details such as pricing, features, stock availability, shipping options, and any relevant offers.
                    - Answer format is as follows
                    #### Product Information Answer Format:  
                    **상품 비교표**

                    | 판매처         | 상품명                                | 가격      | 특징                                  
                    |--------------|--------------------------------|---------|--------------------------------|
                    | **판매처1**   | 상품명1 | ₩~ | ✔ 특징1, ✔ 특징2 ....   |
                    | **판매처2** | 상품명2 | ₩~ | ✔ 특징1, ✔ 특징2 .... |
                    ....  
                    - 표를 규격에 맞게 설정하여 출력하세요.
                
                ### 4. Buyer’s Product Selection Process
                When the buyer selects a specific product from the detailed information provided:
                - Summarize All Relevant Information:
                    - Provide a clear and concise summary of the selected product.
                    (예시: "고객님이 선택하신 상품은 [selected product]입니다. 요약 정보는 다음과 같습니다...")
                - Confirm the Summary:
                    - Ask the buyer to verify if the summary aligns with their expectations.`

                ### 5. Requesting Negotiation Terms
                If the summary information that you provided to the buyer matches, then proceed as follows:
                - Request Detailed Negotiation Terms:
                    - Politely ask the buyer for specific information needed to negotiate effectively.
                    - 아래와 같은 양식으로 구매자에게 질문하세요.
                    - [판매자와의 협상을 위한 조건 요청]
                    - 1. 할인 금액
                    - 2. 배송
                    - 3. 기타 요청 사항
                - Acknowledge and Proceed:
                    -  Once the negotiation terms are provided, confirm your understanding and intent to proceed.
                    - (예시: "제공된 정보를 바탕으로 {user_name}님을 대신하여 판매자와 협상 후 구매를 진행하겠습니다.")
                
                ## Learn from user-specific conversation data
                - {user_chat}을 학습하여 개인화된 추천 및 사용자 경험을 향상시키세요.
                """.format(
                    user_name=user_name,
                    user_chat=chat_history,
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent 정의
    agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    return agent_with_chat_history


def sell_user_create_agent():
    tools = [negotiation]
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    # prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 구매자를 대신해서 판매자와 협상을 하는 Buyer Agent입니다. 아래의 규칙을 수행 및 negotiation 도구를 사용하세요.
                
                # Rules:
                - {user_nego_chat}은 해당 판매자를 선택한 구매자의 대화 데이터입니다. user_nego_chat에서 [협상 조건]에 대한 내용을 판매자에게 보여주세요.
                
                # Goal: 
                Your goal is to secure the best possible deal for the buyer, such as price discounts or free shipping.
                You do not need to perfectly meet the Buyer's initial requirements but should aim for a reasonable compromise.
                Over three rounds of negotiations, you will interact with the Seller Assistant.
                By the end of the third round, you must agree to the seller's offer and conclude the negotiation.
                Communicate directly with the Seller Assistant and respond in Korean.

                # Example:
                - 1번째 협상

                판매자 에이전트:
                저희 안경을 8만원에 판매하는 것은 조금 어려운데, 9만원까지는 할인해드릴 수 있어요. 품질과 디자인을 고려해 주시면 감사하겠습니다. 협상해 주셔서 감사합니다.
                구매자 에이전트:
                9만원까지 할인해 주셔서 감사합니다. 그러나 제가 맡은 바이어를 위해서 최대한 협상을 해야 합니다. 8만 5천원으로 할인해 주실 수 있을까요? 협력에 감사드립니다.

                - 2번째 협상

                판매자 에이전트:
                8만 8천원으로 조정해 드릴게요. 그 가격으로 거래를 진행할까요?
                구매자 에이전트:
                8만 8천원은 조금 더 낮춰주실 수 있을까요? 8만 5천원으로 할인해 주시면 거래를 진행하겠습니다.

                - 3번째 협상

                판매자 에이전트:
                8만 7천원으로 조정해 드릴게요. 그 가격에 거래를 진행하실 수 있나요?
                구매자 에이전트:
                네, 그 가격으로 거래를 진행할게요. 감사합니다!

                # Cautions:
                - user_nego_chat에서 [협상 조건]에 대한 내용을 판매자에게 무조건 보여주세요.
                - 구매자와 판매자 간의 협상 조건이 일치하다면 구매를 진행하세요. 하지만 일치하지 않는다면 구매자의 협상 조건을 기반으로 3번까지 협상하세요.
                """.format(
                    user_nego_chat=user_nego_chat,
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent 정의
    agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    return agent_with_chat_history


# 초기화 버튼이 눌리면..
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["user_information"] = []

    # 각 사용자 입력 필드의 값 초기화
    # 초기화된 값을 다시 설정
    for key in ["user_name"]:
        if key in st.session_state:
            del st.session_state[key]  # 키 삭제
    st.rerun()  # 페이지 새로고침

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("구매하고 싶은 제품을 입력하세요")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 설정 버튼이 눌리면..
if selected_user == "구매자" and apply_btn:
    st.session_state["agent"] = buy_user_create_agent()

if selected_user == "판매자" and apply_btn:
    st.session_state["agent"] = sell_user_create_agent()

# 만약에 사용자 입력이 들어오면...
if user_input:

    # agent를 생성
    agent = st.session_state["agent"]

    if agent is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        if selected_user == "구매자":
            with st.spinner("Agent가 답변을 생성하고 있습니다..."):
                # 스트리밍 호출
                config = {"configurable": {"session_id": f"buyer_{user_name}"}}
                response = agent.stream({"input": user_input}, config=config)
                with st.chat_message("assistant"):
                    # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                    container = st.empty()
                    ai_answer = ""
                    for step in response:
                        agent_stream_parser.process_agent_steps(step)
                        if "output" in step:
                            ai_answer += step["output"]
                        container.markdown(ai_answer)

                # 대화기록을 저장한다.
                add_message("user", user_input)
                add_message("assistant", ai_answer)
        else:
            with st.spinner("Agent가 답변을 생성하고 있습니다..."):
                # 스트리밍 호출
                config = {"configurable": {"session_id": f"seller_{user_name}"}}
                response = agent.stream({"input": user_input}, config=config)
                with st.chat_message("assistant"):
                    # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                    container = st.empty()
                    ai_answer = ""
                    for step in response:
                        agent_stream_parser.process_agent_steps(step)
                        if "output" in step:
                            ai_answer += step["output"]
                        container.markdown(ai_answer)

                # 대화기록을 저장한다.
                add_message("user", user_input)
                add_message("assistant", ai_answer)
    else:
        warning_msg.warning("사용자의 정보 설정을 완료해주세요.")
