from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import dotenv
import os 
dotenv.load_dotenv()

# API 키가 None인 경우 빈 문자열로 설정하여 TypeError 방지
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] ="sk-proj-...."
model = ChatOpenAI(model="gpt-4o")
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
server_params = StdioServerParameters(
    command="uv",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["run", "./mcp_server.py"],
)

async def run():
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                ##### AGENT #####
                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools)

                ##### REQUEST & REPOND #####
                user_input = input("질문을 입력하세요: ")

                print("=====PROMPT=====")
                prompts = await load_mcp_prompt(
                    session, "default_prompt", arguments={"message": user_input}
                )
                print("prompts : ", prompts)
                response = await agent.ainvoke({"messages": prompts})
                # response = await agent.ainvoke({"messages": user_input})

                # print(response)
                print("=====RESPONSE=====")
                print(response["messages"][-1].content)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 명시적으로 잠깐 대기하여 정리 시간 확보
        loop = asyncio.get_event_loop()
        await asyncio.sleep(0.1)
        loop.run_until_complete(loop.shutdown_asyncgens())


import asyncio

asyncio.run(run())