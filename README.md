# 实时联网 AI 助手

基于 LangChain Agent 的智能对话应用，可自动调用搜索引擎获取实时信息。

## 功能
- 多轮对话，带记忆功能
- Agent 自动判断是否需要搜索
- 接入 Tavily 搜索引擎，可查询天气、新闻等实时信息

## 技术栈
- Python
- Gradio（Web 界面）
- LangChain（Agent 框架）
- 智谱 GLM-4-Flash（大模型）
- Tavily Search API（搜索引擎）

## 运行方法
1. 安装依赖：`pip install gradio langchain langchain-openai tavily-python`
2. 替换代码中的 API Key
3. 运行：`python app.py`
4. 打开浏览器访问 http://127.0.0.1:7860
