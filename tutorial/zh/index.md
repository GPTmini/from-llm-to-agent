# 前言

一套从调用大模型 API 出发，用 Python 一步步搭建智能体（Agent）的教程。

不依赖 LangChain、LangGraph 等智能体框架。对话循环、记忆、工具调用、MCP、RAG、长期记忆、推理、规划、反思、多智能体、安全、可靠性，全部从零实现，并配有逐步讲解的 Jupyter Notebook。

全书围绕一个例子展开：学生小明在读这本书，我们为他打造一个**学习助手**。从第 1 章只会聊天的几行代码开始，每章给它加一项新本领：记住对话、查日期、连接远程工具、读懂这本书、记住小明、先想后做、和其他智能体分工合作……到第 24 章，它成为一个能干、安全、可靠的智能体。

* 代码仓库：[gptmini/from-llm-to-agent](https://github.com/gptmini/from-llm-to-agent)
* 许可证：[MIT License](https://github.com/gptmini/from-llm-to-agent/blob/main/LICENSE)

## 特点

* **零框架依赖**：只用大模型厂商的官方 SDK（OpenAI、Anthropic、MCP）和 NumPy，看清智能体内部的每一个环节；
* **循序渐进**：从一次 API 调用，到工具、RAG、推理、规划、多智能体，每章只引入少量新概念；
* **一个贯穿全书的例子**：所有示例都服务于同一个学习助手，学完全书，也就完成了一个完整的小项目；
* **讲解与代码同在**：每章都是一个可以独立运行的 Notebook，包含到这一章为止的全部代码；
* **两家模型对照**：同时支持 OpenAI 和 Claude，对比不同厂商接口的异同。

## 目录

教程共 24 章，分为六个部分，外加一篇技术综述。

目录：`tutorial/zh`

| 目录                            | 章节    | 内容                                          |
|-------------------------------|-------|---------------------------------------------|
| `01-llm-api-and-agent-loop`   | 01–04 | 调用大模型 API、统一不同厂商的接口、消息、最简单的智能体对话循环          |
| `02-conversation-and-memory`  | 05–08 | 对话记忆、摘要压缩、持久化、流式输出                          |
| `03-function-calling-and-mcp` | 09–12 | 工具调用、工具循环、工具箱和 `@tool` 装饰器、MCP                |
| `04-rag`                      | 13–16 | 向量嵌入、切块和索引、检索增强生成（RAG）、长期记忆                 |
| `05-reasoning-and-planning`   | 17–20 | 思维链和 ReAct、推理模型、结构化输出和规划、反思                  |
| `06-multi-agent`              | 21–24 | 工作流、多智能体（子智能体、对话、交接）、安全、可靠性                 |
| `06-multi-agent`              | 综述    | 智能体技术综述：本教程和业界最新进展的距离，供进一步学习参考              |

## 快速开始

### 环境要求

* Python 3.10 及以上
* 依赖：`openai`、`anthropic`、`mcp`、`numpy`、`jupyter-book`

### 安装

```bash
git clone https://github.com/gptmini/from-llm-to-agent.git
cd from-llm-to-agent

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install jupyterlab
```

### API 密钥

教程需要调用大模型的 API，所以要先注册账号、申请 API 密钥（第 1、2 章有详细步骤），然后保存到环境变量：

```bash
export OPENAI_API_KEY="你的 OpenAI 密钥"
export ANTHROPIC_API_KEY="你的 Anthropic 密钥"
```

* **OpenAI 是必需的**：大部分示例默认使用 OpenAI 的 `gpt-5.4-mini`，第 13 章开始的向量嵌入也使用 OpenAI 的嵌入模型；
* **Anthropic 是可选的**：Claude 的示例主要出现在第 2、3 章和第 18 章，没有密钥的话，跳过这些单元格即可。

请不要把密钥直接写在代码里，也不要提交到代码仓库。

### 运行教程

在项目根目录运行 `jupyter lab`，然后按编号顺序打开 `tutorial/zh` 下的 Notebook 即可。

每个 Notebook 都包含到这一章为止的全部代码，可以单独运行。有两处例外：

* 第 15 章使用第 14 章生成的知识库（`knowledge/` 目录），请先运行第 14 章；
* 第 12 章连接远程的 MCP 服务器 DeepWiki，需要能够访问外网。

部分章节会在运行时询问你的输入，比如对话中的问题，或者第 23、24 章发送邮件前的确认（输入 `y` 或 `n`）。

### 费用

教程使用的都是便宜的小模型。问一个简单的问题，花费不到一美分；把全书的例子运行一遍，费用通常也很低。第 24 章会教你统计每次运行的词元用量和费用。

## 学习助手的演进

| 章节    | 学习助手学会了什么                         |
|-------|-----------------------------------|
| 01–04 | 调用大模型，统一不同厂商的接口，进行多轮对话            |
| 05–08 | 记住对话，压缩记忆，保存到文件，流式输出              |
| 09–12 | 调用工具，自动生成工具说明，连接 MCP 服务器          |
| 13–16 | 按意思检索资料，根据资料回答，记住小明这个人            |
| 17–20 | 一步步思考，用推理模型，先计划再执行，回头检查           |
| 21–24 | 和其他智能体分工合作，安全地做事，稳定地运行            |

教程的代码刻意保持简单：每个概念都用最少的代码实现，目的是看清智能体的原理，而不是做一个可以直接上线的产品。和业界真正在用的系统相比还有哪些差距，可以阅读最后的技术综述。

## 项目结构

```text
.
├── tutorial/
│   ├── zh/                             # 中文教程（Jupyter Notebook）
│   │   ├── 01-llm-api-and-agent-loop/
│   │   ├── 02-conversation-and-memory/
│   │   ├── 03-function-calling-and-mcp/
│   │   ├── 04-rag/
│   │   ├── 05-reasoning-and-planning/
│   │   ├── 06-multi-agent/
│   │   └── myst.yml                    # 教程站点（MyST / Jupyter Book）配置
│   └── en/                             # 英文教程，目录结构与 zh 相同
├── requirements.txt
└── LICENSE
```

运行教程时，Notebook 所在的目录下会生成一些数据文件，可以随时删除：

| 目录                  | 内容                  | 生成章节  |
|---------------------|---------------------|-------|
| `memories/`         | 保存的对话记忆（JSON）       | 第 7 章起 |
| `knowledge/`        | 学习助手的小知识库（Markdown） | 第 14、15 章 |
| `index/`            | 向量索引                | 第 14、15 章 |
| `long_term_memory/` | 关于用户的长期记忆           | 第 16、24 章 |

## 相关教程

本书是 [从神经元到大模型](https://github.com/gptmini/from-neuron-to-gpt) 的续篇。前一本书只用 NumPy，从单个神经元一步步手搓出 GPT，讲的是大模型**内部**是怎样工作的；这一本书把大模型当作一个现成的部件，讲的是怎样**围绕**大模型搭建智能体。两本书可以独立阅读。

## 致谢与参考

第 12 章使用了 [DeepWiki](https://deepwiki.com) 提供的免费公开 MCP 服务器。

主要参考文献：

* Anthropic，*Building Effective Agents*，2024。
* Wei 等，*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*，2022。
* Yao 等，*ReAct: Synergizing Reasoning and Acting in Language Models*，2022。
* Lewis 等，*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*，2020。
* Shinn 等，*Reflexion: Language Agents with Verbal Reinforcement Learning*，2023。
* Model Context Protocol 规范：[modelcontextprotocol.io](https://modelcontextprotocol.io)。
