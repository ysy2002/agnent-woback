# agnent-woback

## 项目介绍

我构建了一个基于 OpenAI 的自动化工单回复 Agent。它能自动分析用户提交的工单内容，结合知识库与历史案例生成精准的回复建议，并通过置信度评分机制自动完成低风险工单的闭环处理。目前已在公司客服团队（15人）中落地，每日消耗约 300 万 Token，将工单平均处理时效提升了 75%，人工复核率降低至 20% 以下。

## 核心功能

- **智能工单分类**：自动识别工单类型（咨询、投诉、故障、退款等），路由至对应处理流程
- **知识库检索增强**：基于 RAG 架构，实时检索内部知识库与历史工单，生成上下文相关的回复
- **多轮对话管理**：维护工单上下文，支持追问与补充信息的连续对话
- **置信度评分**：对每条生成回复进行可信度打分，高置信度工单自动发送，低置信度转人工审核
- **效果闭环追踪**：记录用户对回复的满意度反馈，持续优化模型提示词与检索策略

## 技术栈

- **LLM**：OpenAI GPT-4o
- **框架**：LangChain + LangGraph
- **向量数据库**：Chroma
- **后端**：Python / FastAPI
- **部署**：Docker + Kubernetes

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY

# 3. 启动服务
uvicorn app.main:app --reload

# 4. 运行测试
pytest tests/
```

### Docker 运行

```bash
cp .env.example .env   # 填写 OPENAI_API_KEY
docker compose up --build
```

### API 示例

```bash
# 处理工单
curl -X POST http://localhost:8000/api/v1/tickets/reply \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T001","user_id":"U001","content":"我想申请退款","history":[]}'

# 提交满意度反馈
curl -X POST http://localhost:8000/api/v1/tickets/feedback \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T001","satisfied":true}'
```

交互式 API 文档：http://localhost:8000/docs

## 落地数据

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 工单平均处理时效 | 4 小时 | 1 小时 |
| 人工处理占比 | 100% | 20% |
| 用户满意度 | 82% | 91% |
| 每日处理工单量 | 500 条 | 2000 条 |
