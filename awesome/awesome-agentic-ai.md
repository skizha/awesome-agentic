# Awesome Agentic AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![Last Updated](https://img.shields.io/badge/last%20updated-February%202026-blue)
![License](https://img.shields.io/badge/license-CC0%201.0-green)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

> A curated list of frameworks, libraries, tools, platforms, protocols, and resources for building agentic AI systems. Last updated: February 2026.

Agentic AI refers to AI systems that can autonomously plan, reason, use tools, and complete complex multi-step tasks with minimal human intervention.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| [v1.0.2](#) | Feb 2026 | Added `VoltAgent` to Frameworks & Orchestration and `awesome-ai-agent-papers` to Other Awesome Lists after reviewing the `VoltAgent` GitHub organization. |
| [v1.0.1](#) | Feb 2026 | Reviewed against GitHub `awesome-ai` topic page; added missing relevant lists in `Other Awesome Lists` (`awesome-ai-coding-tools`, `awesome-claude`, `awesome-github-copilot`, `awesome-ai-agents`). |
| [v1.0.0](#) | Feb 2026 | Initial release. Covers frameworks, multi-agent systems, coding agents, protocols, LLM models, memory/RAG, tool use, observability, low-code builders, local tooling, cloud platforms, benchmarks, research, and learning resources. |

---

## Contents

- [Changelog](#changelog)
- [Frameworks & Orchestration](#frameworks--orchestration)
- [Multi-Agent Systems](#multi-agent-systems)
- [Coding Agents](#coding-agents)
- [Protocols & Standards](#protocols--standards)
- [LLM Models for Agents](#llm-models-for-agents)
- [Memory & Retrieval (RAG)](#memory--retrieval-rag)
- [Tool Use & Function Calling](#tool-use--function-calling)
- [Observability & Evaluation](#observability--evaluation)
- [Visual / Low-Code Builders](#visual--low-code-builders)
- [Local LLM Tooling](#local-llm-tooling)
- [Cloud & Enterprise Platforms](#cloud--enterprise-platforms)
- [Benchmarks & Datasets](#benchmarks--datasets)
- [Research & Papers](#research--papers)
- [Learning Resources](#learning-resources)
- [Other Awesome Lists](#other-awesome-lists)

---

## Frameworks & Orchestration

General-purpose frameworks for building agentic LLM applications.

| Name | Stars | Description |
|------|-------|-------------|
| [LangChain](https://github.com/langchain-ai/langchain) | 123k+ | The OG framework for chaining LLM calls, tool use, memory, and RAG pipelines. |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 25k+ | Stateful, graph-based orchestration for complex multi-step agent workflows. Built on LangChain. |
| [LlamaIndex](https://github.com/run-llama/llama_index) | 46k+ | Framework for connecting LLMs to data sources; strong RAG and agent workflows. |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | 12k+ | Type-safe, validation-first agent framework from the Pydantic team. Ideal for compliance-heavy use cases. |
| [Haystack](https://github.com/deepset-ai/haystack) | 22k+ | End-to-end NLP framework with pipeline-based agent orchestration from deepset. |
| [DSPy](https://github.com/stanfordnlp/dspy) | 24k+ | Stanford's framework for programming LLM pipelines using declarative modules and auto-optimization. |
| [smolagents](https://github.com/huggingface/smolagents) | 25k+ | Hugging Face's minimal, code-first agent framework (~1k lines of core logic). Model- and tool-agnostic. |
| [Mastra](https://github.com/mastra-ai/mastra) | 6k+ | TypeScript-native agentic framework with workflows, memory, and tool support. |
| [VoltAgent](https://github.com/VoltAgent/voltagent) | 6k+ | Open-source TypeScript AI agent framework with workflows, memory, MCP, RAG, guardrails, and production-oriented ops integrations. |
| [Agno](https://github.com/agno-agi/agno) | 22k+ | Production-grade multi-agent framework with strong memory, context, and documentation. (formerly phidata) |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | 26k+ | Microsoft's SDK for integrating LLMs into C#, Python, and Java apps with agentic patterns. |
| [Instructor](https://github.com/jxnl/instructor) | 10k+ | Structured outputs from LLMs via Pydantic schemas. Pairs well with any agent framework. |

---

## Multi-Agent Systems

Frameworks specifically designed for coordinating multiple collaborating agents.

| Name | Stars | Description |
|------|-------|-------------|
| [AutoGen](https://github.com/microsoft/autogen) | 53k+ | Microsoft's multi-agent orchestration platform for collaborative, conversational agents. |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 42k+ | Role-based multi-agent teams with explicit crew definitions and task delegation. |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | 10k+ | OpenAI's production-ready multi-agent SDK with built-in tracing, guardrails, and memory. Replaces the deprecated Swarm. |
| [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 177k+ | The original autonomous agent platform. Goal-driven, self-prompting LLM agent. |
| [AgentGPT](https://github.com/reworkd/AgentGPT) | 32k+ | Browser-based, no-code autonomous agent builder. |
| [BabyAGI](https://github.com/yoheinakajima/babyagi) | 20k+ | Minimal, educational task-driven autonomous agent for research and experimentation. |
| [MetaGPT](https://github.com/geekan/MetaGPT) | 48k+ | Multi-agent framework that assigns GPT agents software engineering roles (PM, dev, QA). |
| [ChatDev](https://github.com/OpenBMB/ChatDev) | 27k+ | Simulates a software company with LLM agents playing different development roles. |

---

## Coding Agents

Agents and tools specifically designed for autonomous software engineering tasks.

| Name | Stars | Description |
|------|-------|-------------|
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | 50k+ | (formerly OpenDevin) Open-source, modular platform for autonomous software agents. MIT licensed. |
| [SWE-agent](https://github.com/princeton-nlp/SWE-agent) | 14k+ | Princeton's agent for autonomously resolving GitHub issues. Powers the SWE-bench benchmark. |
| [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) | 60k+ | Lets LLMs run code locally. Natural language interface to your computer's capabilities. |
| [Aider](https://github.com/paul-gauthier/aider) | 24k+ | AI pair programmer in your terminal. Works with Git and supports most major LLMs. |
| [Cline](https://github.com/cline/cline) | 20k+ | Autonomous coding agent for VS Code with MCP support and browser use. |
| [Cursor](https://cursor.sh) | — | AI-powered IDE with agentic code editing, Composer, and multi-file context. |
| [Windsurf](https://codeium.com/windsurf) | — | Codeium's agentic IDE with Cascade flow engine for collaborative coding. |
| [Devin](https://cognition.ai/devin) | — | Cognition AI's fully autonomous software engineer agent (commercial). |
| [GitHub Copilot](https://github.com/features/copilot) | — | GitHub's AI coding assistant, now with agentic coding mode in IDEs. |

---

## Protocols & Standards

Open protocols enabling interoperability between agents and tools.

| Name | Origin | Purpose |
|------|--------|---------|
| [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) | Anthropic | Universal standard for connecting LLMs/agents to external tools, APIs, and data sources. Vertical (agent ↔ tool). |
| [Agent-to-Agent (A2A)](https://github.com/google/A2A) | Google | Peer-to-peer protocol for agents to coordinate, delegate, and collaborate across vendors. Horizontal (agent ↔ agent). |
| [Agent Communication Protocol (ACP)](https://github.com/i-am-bee/acp) | IBM / BeeAI | RESTful, session-aware protocol for structured agent message exchange. Internal comms. |
| [Agent Network Protocol (ANP)](https://github.com/agent-network-protocol/agentnetworkprotocol) | Community | Decentralized agent discovery and collaboration using W3C DIDs for open networks. |

> **Quick guide**: Use MCP for tool access, A2A for cross-agent teamwork, ACP for intra-cluster messaging, ANP for decentralized agent networks.

---

## LLM Models for Agents

Frontier and open-source models well-suited for agentic workloads (tool use, long-horizon reasoning).

### Frontier / Commercial
| Model | Provider | Notes |
|-------|----------|-------|
| [GPT-5.2](https://platform.openai.com/docs/models) | OpenAI | Current flagship model. Supersedes GPT-4o/o3 (retired Feb 2026). Best-in-class tool calling and reasoning. |
| [Claude Sonnet 4.6 / Opus 4.6](https://www.anthropic.com/claude) | Anthropic | Latest Claude generation (Feb 2026). 1M token context, superior agentic coding and extended thinking. |
| [Gemini 3 Pro / 2.5 Flash](https://deepmind.google/technologies/gemini/) | Google | Gemini 3 Pro is Google's latest flagship; 2.5 Flash remains available for high-speed, multimodal agents. |
| [Grok-3](https://x.ai/grok) | xAI | Strong coding and reasoning capabilities. |

### Open Source
| Model | GitHub | Notes |
|-------|--------|-------|
| [DeepSeek-V3 / R1](https://github.com/deepseek-ai/DeepSeek-V3) | deepseek-ai | State-of-the-art open model; cost-effective for agentic workflows at scale. |
| [Qwen 3](https://github.com/QwenLM/Qwen3) | Alibaba / QwenLM | Latest generation; top-tier multilingual, math, and code-capable open model (Apache 2.0). |
| [Mistral / Mixtral](https://github.com/mistralai) | Mistral AI | Efficient MoE models; good balance of quality and cost. |
| [Llama 4](https://github.com/meta-llama/llama-models) | Meta | Meta's latest open family with 10M token context; strong reasoning and coding for agentic use. |
| [Phi-4](https://huggingface.co/microsoft/phi-4) | Microsoft | Small but powerful; great for edge/local agent deployments. |
| [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) | Moonshot AI | High-performing open model for tool use and agentic action completion. |

---

## Memory & Retrieval (RAG)

Libraries and services for giving agents persistent memory and knowledge retrieval.

| Name | Stars | Description |
|------|-------|-------------|
| [Mem0](https://github.com/mem0ai/mem0) | 28k+ | Intelligent memory layer for AI agents. Stores, updates, and retrieves memories across sessions. |
| [LangMem](https://github.com/langchain-ai/langmem) | 2k+ | LangChain's native long-term memory SDK for agent state across conversations. |
| [Chroma](https://github.com/chroma-core/chroma) | 18k+ | Open-source, developer-friendly vector database for embeddings and RAG. |
| [Weaviate](https://github.com/weaviate/weaviate) | 13k+ | Open-source vector search engine with hybrid (vector + keyword) search. |
| [Qdrant](https://github.com/qdrant/qdrant) | 22k+ | High-performance vector similarity search, written in Rust. |
| [Milvus](https://github.com/milvus-io/milvus) | 33k+ | Scalable, cloud-native vector database built for billion-scale embeddings. |
| [pgvector](https://github.com/pgvector/pgvector) | 14k+ | Open-source vector extension for PostgreSQL. Simple and production-ready. |
| [Zep](https://github.com/getzep/zep) | 3k+ | Long-term memory store and knowledge graph for LLM agents. |

---

## Tool Use & Function Calling

Libraries that help agents discover, call, and manage external tools and APIs.

| Name | Stars | Description |
|------|-------|-------------|
| [Composio](https://github.com/ComposioHQ/composio) | 17k+ | 250+ pre-built tool integrations (GitHub, Slack, Gmail, etc.) for AI agents. |
| [Browserbase](https://github.com/browserbase/browserbase) | 1k+ | Cloud browser infrastructure for headless browser automation in agents. |
| [Browser Use](https://github.com/browser-use/browser-use) | 60k+ | Library enabling LLM agents to control and interact with web browsers. |
| [Playwright](https://github.com/microsoft/playwright) | 70k+ | Microsoft's browser automation library widely used for agent web interactions. |
| [Toolhouse](https://github.com/toolhouseai/toolhouse-sdk-python) | 1k+ | Cloud-hosted tool execution layer for agents with built-in auth and caching. |

---

## Observability & Evaluation

Monitor, trace, debug, and evaluate your agent pipelines in development and production.

| Name | Stars | Open Source | Description |
|------|-------|-------------|-------------|
| [Langfuse](https://github.com/langfuse/langfuse) | 10k+ | ✅ | Deep LLM tracing and observability; self-hostable. Integrates with most frameworks. |
| [LangSmith](https://smith.langchain.com) | — | ❌ | LangChain's hosted tracing, debugging, and dataset curation platform. Minimal overhead. |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | 5k+ | ✅ | OpenTelemetry-native agent tracing and evaluation for enterprise ML teams. |
| [Weave](https://github.com/wandb/weave) | 2k+ | ✅ | W&B's tool for tracking, evaluating, and iterating on LLM/agent pipelines. |
| [Helicone](https://github.com/Helicone/helicone) | 3k+ | ✅ | Lightweight API monitoring for token cost, latency, and caching. |
| [AgentOps](https://github.com/AgentOps-AI/agentops) | 3k+ | ✅ | Agent session replay, cost tracking, and error detection. |
| [Braintrust](https://www.braintrust.dev) | — | ❌ | Enterprise eval platform with fine-grained scoring for agent outputs. |
| [Maxim AI](https://www.getmaxim.ai) | — | ❌ | End-to-end simulation, evaluation, and observability for agentic systems. |
| [RAGAS](https://github.com/explodinggradients/ragas) | 8k+ | ✅ | Evaluation framework specifically for RAG pipelines—faithfulness, relevance, etc. |

---

## Visual / Low-Code Builders

Build agent workflows visually without deep coding required.

| Name | Stars | Description |
|------|-------|-------------|
| [Langflow](https://github.com/langflow-ai/langflow) | 55k+ | Drag-and-drop visual builder for LangChain-based agent pipelines. |
| [Flowise](https://github.com/FlowiseAI/Flowise) | 40k+ | No-code LLM flow builder with prebuilt nodes for chains, agents, and tools. |
| [Dify](https://github.com/langgenius/dify) | 80k+ | Open-source LLM app development platform with prompt IDE, RAG, and agent tools. |
| [n8n](https://github.com/n8n-io/n8n) | 90k+ | Workflow automation tool with native AI/LLM nodes for agentic task pipelines. |
| [Rivet](https://github.com/Ironclad/rivet) | 3k+ | Open-source AI agent runtime and visual programming environment by Ironclad. |
| [PromptFlow](https://github.com/microsoft/promptflow) | 10k+ | Microsoft's toolkit for building, evaluating, and deploying LLM flows. |

---

## Local LLM Tooling

Run and manage LLMs locally for private, offline, or resource-constrained agent deployments.

| Name | Stars | Description |
|------|-------|-------------|
| [Ollama](https://github.com/ollama/ollama) | 165k+ | The easiest way to run LLMs locally on Mac, Windows, and Linux. |
| [LM Studio](https://lmstudio.ai) | — | Desktop app for discovering, downloading, and running local LLMs with a GUI. |
| [vLLM](https://github.com/vllm-project/vllm) | 50k+ | Fast, memory-efficient LLM inference server. Production-ready local hosting. |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 75k+ | C/C++ inference for LLaMA models; runs on CPU and GPU with quantization. |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | 72k+ | Open-source ecosystem for running powerful LLMs locally on consumer hardware. |
| [Jan](https://github.com/janhq/jan) | 26k+ | Open-source offline-first personal AI assistant with local model support. |

---

## Cloud & Enterprise Platforms

Managed platforms and cloud offerings for deploying production agentic AI systems.

| Name | Provider | Description |
|------|----------|-------------|
| [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/) | AWS | Fully managed agent builder with tool use, knowledge bases, and guardrails. |
| [Azure AI Agent Service](https://azure.microsoft.com/en-us/products/ai-services/ai-agent-service) | Microsoft | Managed agents on Azure with AutoGen orchestration and enterprise security. |
| [Google Vertex AI Agents](https://cloud.google.com/products/agent-builder) | Google Cloud | Build, deploy, and manage agents using Gemini models on GCP. |
| [Salesforce Agentforce](https://www.salesforce.com/agentforce/) | Salesforce | Enterprise autonomous agents for CRM, customer service, and business workflows. |
| [ServiceNow AI Agents](https://www.servicenow.com/products/ai-agents.html) | ServiceNow | Domain-specific agents for IT, HR, and customer support automation. |
| [Relevance AI](https://relevanceai.com) | Relevance AI | Visual multi-agent workflow orchestration for enterprise teams. |
| [Cognitiev / E2B](https://e2b.dev) | E2B | Sandboxed code execution infrastructure for running agent-generated code safely. |

---

## Benchmarks & Datasets

Evaluate agent capabilities on standardized tasks.

| Name | Description |
|------|-------------|
| [SWE-bench](https://github.com/princeton-nlp/SWE-bench) | Evaluates agents on resolving real GitHub issues across popular Python repos. |
| [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | General AI Assistants benchmark requiring real-world reasoning and tool use. |
| [AgentBench](https://github.com/THUDM/AgentBench) | Multi-environment benchmark covering web, DB, OS, and game environments. |
| [WebArena](https://github.com/web-arena-x/webarena) | Realistic web browser environment benchmark for autonomous web agents. |
| [TAU-bench](https://github.com/sierra-research/tau-bench) | Task-oriented benchmark for evaluating LLM agents in real-world API tool use. |
| [BrowsingBench](https://browsingbench.github.io) | Evaluates agents on complex multi-step web browsing tasks. |
| [OSWorld](https://github.com/xlang-ai/OSWorld) | Benchmark for desktop GUI agents completing OS-level tasks. |

---

## Research & Papers

Seminal papers and surveys shaping agentic AI research.

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al. (2022) — Foundational ReAct prompting pattern for agents.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Schick et al. (2023)
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580) — Shen et al. (2023)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) — Wang et al. (2023)
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) — Liu et al. (2023)
- [OpenHands: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741) — Wang et al. (2024)
- [A Survey of Agent Interoperability Protocols: MCP, A2A, ACP, ANP](https://arxiv.org/abs/2505.02279) — 2025 survey
- [Agentic AI Frameworks: Architectures, Protocols, and Design Challenges](https://arxiv.org/abs/2508.10146) — 2025 survey

---

## Learning Resources

### Courses & Tutorials
- [DeepLearning.AI: AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — Hands-on short course.
- [DeepLearning.AI: Multi AI Agent Systems with CrewAI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai)
- [DeepLearning.AI: Building Agentic RAG with LlamaIndex](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex)
- [Hugging Face: Agents Course](https://huggingface.co/learn/agents-course/) — Free, open-source course on building AI agents.
- [LangChain Academy: Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph)

### Blogs & Newsletters
- [Lilian Weng's Blog](https://lilianweng.github.io) — Deeply technical posts on LLM agents and planning.
- [Simon Willison's Weblog](https://simonwillison.net) — Practical notes on LLMs, tools, and agents.
- [The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/) — Weekly AI newsletter.
- [DAIR.AI Newsletter](https://github.com/dair-ai/ML-Papers-of-the-Week) — Top ML papers weekly.
- [Ahead of AI](https://magazine.sebastianraschka.com) — Sebastian Raschka's newsletter on LLMs and agents.

### YouTube Channels
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) — Deep dives into LLMs and neural networks.
- [AI Jason](https://www.youtube.com/@AIJasonZ) — Practical tutorials on AI agents and tools.
- [Sam Witteveen](https://www.youtube.com/@samwitteveenai) — LangChain, LLM agents, and RAG tutorials.

---

## Other Awesome Lists

- [awesome-ai-coding-tools](https://github.com/ai-for-developers/awesome-ai-coding-tools) — Curated list of AI-powered coding tools.
- [awesome-claude](https://github.com/tonysurfly/awesome-claude) — Curated list of Claude ecosystem tools and resources (including coding-agent workflows).
- [awesome-github-copilot](https://github.com/anchildress1/awesome-github-copilot) — Prompts, agents, skills, and practical resources for GitHub Copilot.
- [awesome-ai-agents (NipunaRanasinghe)](https://github.com/NipunaRanasinghe/awesome-ai-agents) — Curated frameworks, tools, and resources for AI agents.
- [awesome-ai-agent-papers](https://github.com/VoltAgent/awesome-ai-agent-papers) — Curated collection of recent AI agent research papers (agent engineering, memory, evals, workflows).
- [awesome-ai-agents](https://github.com/slavakurilyak/awesome-ai-agents) — 300+ agentic AI projects with metrics and categories.
- [awesome-llm-agents](https://github.com/kaushikb11/awesome-llm-agents) — Curated LLM-powered agent papers and projects.
- [awesome-langchain](https://github.com/kyrolabs/awesome-langchain) — Tools and projects built with LangChain.
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) — Curated list of MCP servers for Claude and other agents.
- [awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts) — Prompt engineering resource for LLMs.
- [LLM-Agents-Papers](https://github.com/AGI-Edgerunners/LLM-Agents-Papers) — Comprehensive list of agent research papers.
- [Awesome AI Agents (korchasa)](https://awesome-ai-agents.korchasa.dev/) — Auto-maintained directory of multi-agent systems.

---

## Contributing

Pull requests are welcome! Please ensure entries are:
- Actively maintained (last commit within 12 months, or historically significant)
- Relevant to agentic AI (autonomous decision-making, tool use, planning, multi-agent)
- Accompanied by a clear, one-line description

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This list is released under the [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) public domain dedication.

---

*Inspired by the [awesome](https://github.com/sindresorhus/awesome) list format.*
