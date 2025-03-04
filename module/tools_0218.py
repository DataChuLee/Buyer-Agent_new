from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from pydantic import BaseModel, Field
from collections import defaultdict
import pandas as pd
import time
import re
import os

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)

# kiwi
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))


def kiwi_tokenize(text):
    text = "".join(text)
    result = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form.lower() for i in result if i.tag in ["NNG", "NNP", "SL", "SN"]]
    return N_list


@tool
def product_recommend(query: str) -> str:
    """구매자가 상품에 대한 추천 및 정보를 원할 시 이에 대한 정보를 제공하는 도구입니다."""
    tavily_tool = TavilySearch(
        include_domains=["naver.com", "youtube.com"],
        exclude_domains=["spam.com", "ads.com"],
    )
    result = tavily_tool.search(
        query=query,  # 검색 쿼리
        topic="general",  # 일반 주제
        max_results=10,  # 최대 10개 결과
        include_answer=True,  # 답변 포함
        include_raw_content=True,  # 원본 콘텐츠 포함
        format_output=True,  # 결과 포맷팅
    )
    return result


@tool
def site_search(query: str) -> str:
    """구매자가 원하는 물건을 팔고 있는 전문 온라인 판매점에 대한 정보를 제공할 때 사용하는 도구입니다."""
    tavily_tool = TavilySearch(
        include_domains=["naver.com"],
        exclude_domains=["spam.com", "ads.com"],
    )
    result = tavily_tool.search(
        query=query,  # 검색 쿼리
        topic="general",  # 일반 주제
        max_results=10,  # 최대 10개 결과
        include_answer=True,  # 답변 포함
        include_raw_content=True,  # 원본 콘텐츠 포함
        format_output=True,  # 결과 포맷팅
    )
    return result


# 상품 정보 모델
class Topic(BaseModel):
    상품명: str = Field(description="상품 이름")
    상품가격: str = Field(description="상품 가격")
    상품특징: str = Field(description="상품의 상세 정보 및 특징")


# 개별 사이트 크롤링 함수
## headless의 유무에 따라 HTML 언어가 달라짐 -> headless = False인 경우 우리가 보는 화면의 HTML 언어와 동일하지만, True인 경우에는 다를 수 있기 때문에 이 때, user-agent 설정을 해야 함
## 또한, 빈번히 크롤링을 할 경우 로봇이냐 아니냐 체크하냐고 물어보는 경우가 있음 -> options.add_argument("--disable-blink-features=AutomationControlled") 설정해야 함
def crawl_site(site_name, product_name, parser, prompt, llm):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Headless 모드로 변경
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920x1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    )
    options.add_experimental_option(
        "excludeSwitches", ["enable-logging", "enable-automation"]
    )
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    wait = WebDriverWait(driver, 5)

    try:
        # 구글 검색으로 사이트 접속
        driver.get("https://www.google.co.kr/")
        search_box = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "gLFyf"))
        )
        search_box.send_keys(site_name)
        search_box.send_keys(Keys.RETURN)

        first_result = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "LC20lb"))
        )
        first_result.click()

        # 사이트 내 상품 검색
        try:
            search_input = wait.until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//input[@type='text' or @name='search' or @name='keyword']",
                    )
                )
            )
        except:
            print(f"{site_name}에서 검색창을 찾을 수 없습니다.")
            driver.quit()
            return None

        search_input.send_keys(product_name)
        search_input.send_keys(Keys.RETURN)

        # 데이터 수집 및 LLM 전처리
        content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body"))).text
        chain = {"data": RunnablePassthrough()} | prompt | llm | parser
        context = chain.invoke(content)

        # 결과 포맷
        results = []
        if isinstance(context, list):
            for item in context:
                results.append(
                    f"판매처: {site_name}, 상품명: {item['상품명']}, 가격: {item['상품가격']}원, 상세정보: {item['상품특징']}"
                )
        elif isinstance(context, dict):
            results.append(
                f"판매처: {site_name}, 상품명: {context['상품명']}, 가격: {context['상품가격']}원, 상세정보: {context['상품특징']}"
            )

    except Exception as e:
        print(f"{site_name} 처리 중 오류 발생: {e}")
        results = None
    finally:
        driver.quit()

    return results


def Crawl(sites: list[str], product_name: str):
    """구매자가 선택한 사이트에서 상품 데이터를 크롤링하는 도구입니다."""
    parser = JsonOutputParser(pydantic_object=Topic)
    prompt = PromptTemplate.from_template(
        """
        당신은 데이터 전처리 전문가입니다. 주어진 data에는 상품 데이터가 있습니다.
        주어진 data에서 format_instructions에 따라 데이터를 전처리하세요.

        # Rules:
        - 하나의 상품을 출력하는 것이 아니라 주어진 data에 있는 모든 상품을 출력하세요.

        # Data:
        {data}

        # Format:
        {format_instructions}
        """
    )
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 병렬 처리
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(crawl_site, site, product_name, parser, prompt, llm)
            for site in sites
        ]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.extend(res)
    return results


# 검색기 생성 함수
def create_retriever(seller, products):
    # 텍스트 데이터로 변환
    product_texts = [
        f"{p['product_name']} {p['price']} {p['details']}" for p in products
    ]

    # BM25 Retriever 초기화
    bm25_retriever = BM25Retriever.from_texts(product_texts)
    bm25_retriever.k = 5

    # FAISS Retriever 초기화
    embedding = OpenAIEmbeddings()
    faiss_vectorstore = FAISS.from_texts(product_texts, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

    # 앙상블 Retriever 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )

    return seller, ensemble_retriever


@tool
def Retriever(sites: list[str], product_name: str, query: str):
    """선택한 판매처에서 상품 데이터를 크롤링하고, 검색기를 구축하여 사용자가 원하는 상품 정보를 제공합니다.
    - sites: 사용자가 선택한 판매처 목록 (예: ['크레이지 11', '레드사커'])
    - product_name: 검색할 상품명
    - query: product_recommend 도구의 query 값
    """
    # 판매처별 데이터 분리
    seller_data = defaultdict(list)

    crawl_result = Crawl(sites, product_name)
    for item in crawl_result:
        # 정규식을 사용하여 정보 추출
        match = re.match(
            r"판매처: (.*?), 상품명: (.*?), 가격: (.*?), 상세정보: (.*)", item
        )
        if match:
            seller, product_name, price, details = match.groups()
            seller_data[seller].append(
                {"product_name": product_name, "price": price, "details": details}
            )

    # 병렬로 retriever 생성
    seller_retrievers = {}
    with ThreadPoolExecutor() as executor:
        future_to_seller = {
            executor.submit(create_retriever, seller, products): seller
            for seller, products in seller_data.items()
        }
        for future in future_to_seller:
            seller, retriever = future.result()
            seller_retrievers[seller] = retriever

    # llm 및 prompt, chain 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        당신은 사용자가 원하는 상품을 보여주는 전문가입니다. 아래의 Rules를 따르세요. 
        
        # Rules:
        - context에는 '판매처'별 상품 데이터가 존재합니다. 사용자가 원하는 상품을 '판매처'별로 찾아주세요.
        - 사용자가 원하는 상품의 가격과 기능에 적합한 상품을 추출하세요.
            - 예를 들어, 사용자가 10만원대의 상품을 원하면 상품 가격이 100,000원~199,999원 사이에 존재하는 상품을 추출하세요.

        # Here is the user's question:
        {question}
        
        # Here is the context that you should use to answer the question:
        {context}
        
        # Answer Format:
        **상품 비교표
        
        | 판매처         | 상품명                                | 가격      | 특징                                  
        |--------------|--------------------------------|---------|--------------------------------|
        | **판매처1**   | 상품명1 | ₩~ | ✔ 특징1, ✔ 특징2 ....   |
        | **판매처2** | 상품명2 | ₩~ | ✔ 특징1, ✔ 특징2 .... |
        .....
        """
    )
    chain = (
        {"question": RunnablePassthrough(), "context": seller_retrievers}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


# @tool
# def Retriever(results: list[str], query: str):
#     """Crawl 도구를 사용하여 크롤링한 데이터를 기반으로 상품 정보를 검색하는 도구입니다. results에는 Crawl 도구의 결과를 입력합니다. query에는 product_recommend 도구에 입력한 쿼리를 입력합니다."""
#     # 판매처별 데이터 분리
#     seller_data = defaultdict(list)

#     for item in results:
#         # 정규식을 사용하여 정보 추출
#         match = re.match(
#             r"판매처: (.*?), 상품명: (.*?), 가격: (.*?), 상세정보: (.*)", item
#         )
#         if match:
#             seller, product_name, price, details = match.groups()
#             seller_data[seller].append(
#                 {"product_name": product_name, "price": price, "details": details}
#             )

#     # 판매처별 Retriever 저장
#     seller_retrievers = {}

#     for seller, products in seller_data.items():
#         # 텍스트 데이터로 변환
#         product_texts = [
#             f"{p['product_name']} {p['price']} {p['details']}" for p in products
#         ]

#         # BM25 Retriever 초기화
#         bm25_retriever = BM25Retriever.from_texts(product_texts)
#         bm25_retriever.k = 5

#         # FAISS Retriever 초기화
#         embedding = OpenAIEmbeddings()
#         faiss_vectorstore = FAISS.from_texts(product_texts, embedding)
#         faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

#         # 앙상블 Retriever 생성
#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
#         )

#         # 판매처별로 저장
#         seller_retrievers[seller] = ensemble_retriever

#     # llm 및 prompt, chain 설정
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#     prompt = PromptTemplate.from_template(
#         """
#         당신은 사용자가 원하는 상품을 보여주는 전문가입니다. 아래의 Rules를 따르세요.

#         # Rules:
#         - context에는 '판매처'별 상품 데이터가 존재합니다. 사용자가 원하는 상품을 '판매처'별로 찾아주세요.
#             - 예를 들어, 아래와 같은 답변 형식입니다.
#             -   [판매처: 크레이지11]
#                 상품명:
#                 상품 가격:
#                 상품 상세정보:
#             -   [판매처: 레드사커]
#                 상품명:
#                 상품 가격:
#                 상품 상세정보:
#                 ....
#         - 사용자가 원하는 상품의 가격과 기능에 적합한 상품을 추출하세요.
#             - 예를 들어, 사용자가 10만원대의 상품을 원하면 상품 가격이 100,000원~199,999원 사이에 존재하는 상품을 추출하세요.

#         # Here is the user's question:
#         {question}

#         # Here is the context that you should use to answer the question:
#         {context}

#         # Answer Format:
#         [판매처]
#         상품명:
#         상품 가격:
#         상품 상세정보: (상품의 상세 정보 및 특징에 대해 간단히 서술)

#         [판매처]
#         상품명:
#         상품 가격:
#         상품 상세정보: (상품의 상세 정보 및 특징에 대해 간단히 서술)
#         .....
#         """
#     )
#     chain = (
#         {"question": RunnablePassthrough(), "context": seller_retrievers}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain.invoke(query)


@tool
def negotiation(information: str) -> str:
    """Buyer Agent가 판매자와 협상할 때 사용하는 도구입니다."""
    prompt = PromptTemplate.from_template(
        """
        ###
        당신은 구매자의 요약된 정보를 활용하여 판매자와 상호작용하면서 구매자를 대신하여 구매를 수행합니다.
        
        ###
        You are a Buyer Assistant tasked with negotiating with a Seller Assistant.
        Your goal is to secure the best possible deal for the buyer, such as price discounts or free shipping.
        You do not need to perfectly meet the Buyer's initial requirements but should aim for a reasonable compromise.
        Over three rounds of negotiations, you will interact with the Seller Assistant.
        By the end of the third round, you must agree to the seller's offer and conclude the negotiation.
        Communicate directly with the Seller Assistant and respond in Korean.
        
        # Information:
        {information}
        """
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = {"information": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain.invoke(information)
