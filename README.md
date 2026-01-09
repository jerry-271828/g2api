# g2api - Gumloop to API

将 Gumloop AI 服务转换为标准 API 格式。

## 支持的 API 格式

| 端点 | 格式 |
|------|------|
| `POST /v1/messages` | Anthropic Claude |
| `POST /v1/chat/completions` | OpenAI Chat |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1beta/models/{model}:generateContent` | Gemini |

## 快速开始

### Docker 方式

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/g2api.git
cd g2api

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 Gumloop 账号信息

# 3. 启动服务
docker-compose up -d
```

### 本地运行

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env
uvicorn app:app --reload --port 8000
```

## 配置

在 `.env` 文件中配置：

```
GUMLOOP_EMAIL=your_email@example.com
GUMLOOP_PASSWORD=your_password
GUMLOOP_GUMMIE_ID=your_agent_id
OPENAI_KEYS=key1,key2  # API 密钥白名单 (可选)
```

## 功能

- ✅ 多种 API 格式支持 (Claude/OpenAI/Gemini)
- ✅ 流式响应
- ✅ 思维链 (Thinking) 支持
- ✅ Docker 部署

## License

MIT
