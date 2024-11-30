from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini")
