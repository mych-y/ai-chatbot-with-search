# ============================================
# 项目：基于 LangChain 的实时联网 AI 助手
# 功能：多轮对话、Agent 自主调用搜索工具获取实时信息
# ============================================

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient

# ============================================
#  配置 API Key
# ============================================
# 智谱 AI Key
ZHIPU_API_KEY = "智谱密钥"
# Tavily Key
TAVILY_API_KEY = "Tavily密钥"

# ============================================
#  初始化模型和搜索客户端
#  使用智谱 GLM-4-Flash 模型，通过 OpenAI 兼容接口
# ============================================
chat = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ============================================
#  定义搜索工具
# ============================================
@tool
def search_web(query: str) -> str:
    """当需要查找实时信息、新闻、天气、百科知识时使用这个工具。"""
    try:
        response = tavily_client.search(
            query=query,
            max_results=3,
            include_answer=True
        )
        answer = response.get("answer", "")
        results = response.get("results", [])

        # 优先返回 Tavily 自动生成的摘要
        if answer:
            return f"实时信息摘要：{answer}"

        # 如果没有摘要，返回前两条详细结果
        formatted = ""
        for i, r in enumerate(results[:2]):
            title = r.get('title', '')
            content = r.get('content', '')[:150]
            formatted += f"{title}: {content}\n"
        return formatted if formatted else f"未找到关于'{query}'的实时信息。"
    except Exception as e:
        return f"搜索时出错：{str(e)}"

# ============================================
# Agent 会自动判断何时调用工具、何时直接回答
# ============================================
tools = [search_web]
agent = create_agent(model=chat, tools=tools)

# ============================================
# 对话记忆与核心逻辑
# ============================================
messages_history = []

def respond(message, chat_history):
    global messages_history

    # 第一次对话时，强制塞入系统指令，必须使用搜索工具
    if not messages_history:
        messages_history.append((
            "system",
            "你是一个必须使用搜索工具的助手。对于任何询问实时信息（天气、新闻、事件、股价等）的问题，你必须调用 search_web 工具来获取最新信息，绝不要使用你自己的内部知识。搜索时使用简洁的关键词。如果你不确定某个信息是否需要搜索，那就先搜索一下。"
        ))


    messages_history.append(("user", message))

    # 调用 Agent
    result = agent.invoke({"messages": messages_history})

    # 提取 Agent 最后一条 AI 回答
    ai_answer = None
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            ai_answer = msg.content
            break

    if ai_answer is None:
        ai_answer = "抱歉，我处理这个问题时出了点问题。"

    # 更新历史
    messages_history.append(("ai", ai_answer))

    # 更新 Gradio 聊天界面
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ai_answer})

    return "", chat_history

def clear_all():
    global messages_history
    messages_history = []
    return None

# ============================================
# 6. Gradio 界面
# ============================================
with gr.Blocks(title="我的AI助手（联网搜索）") as demo:
    gr.Markdown("# 🤖 我的AI助手（实时联网搜索）")
    chatbot = gr.Chatbot(label="对话区", height=400)
    msg = gr.Textbox(label="输入你的问题", placeholder="试试问'今天保定天气怎么样'或'最新AI新闻'")
    clear = gr.Button("清空对话")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_all, None, chatbot)


if __name__ == "__main__":
    demo.launch()
    