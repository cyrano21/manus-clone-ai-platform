# Manus Clone - Integrated AI Platform

A comprehensive full-stack platform featuring document editing, contextual AI chat, autonomous agents with browser automation, RAG capabilities, and AI-powered media generation.

## 🏗️ Architecture

This is a monorepo containing:

- **`apps/web`** - Next.js 14+ frontend with App Router and Server Components
- **`apps/workers`** - Node.js TypeScript workers for background processing
  - `browser-runner` - Playwright-based browser automation
  - `rag-worker` - Document ingestion and vector search
  - `metering-worker` - Usage tracking and billing
- **`services/python`** - FastAPI services for AI capabilities
  - `images` - Image generation using Stable Diffusion
  - `video` - Video generation using Stable Video Diffusion
- **`packages/shared`** - Shared TypeScript types and utilities
- **`prisma`** - Database schema and migrations

## 🚀 Quick Start

### Prerequisites

- Node.js 20+
- pnpm 8+
- Docker & Docker Compose
- Python 3.11+ (for AI services)

### 1. Clone and Install

```bash
git clone <repository-url>
cd manus-clone
pnpm install
```

### 2. Environment Setup

```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

### 3. Start Infrastructure

```bash
# Start all services (PostgreSQL, Redis, Qdrant, MinIO, Ollama, AI services)
pnpm docker:up

# Wait for services to be ready (check logs)
pnpm docker:logs
```

### 4. Database Setup

```bash
# Run migrations
pnpm db:migrate

# Seed database (optional)
pnpm db:seed
```

### 5. Start Development

```bash
# Start all applications and workers
pnpm dev
```

The application will be available at http://localhost:8000

## 🛠️ Development

### Project Structure

```
manus-clone/
├── apps/
│   ├── web/                    # Next.js frontend
│   │   ├── src/app/           # App Router pages
│   │   ├── src/components/    # React components
│   │   └── src/lib/          # Utilities
│   └── workers/               # Background workers
│       ├── browser-runner/    # Playwright automation
│       ├── rag-worker/       # Document processing
│       └── metering-worker/  # Usage tracking
├── services/python/           # AI services
│   ├── images/               # Image generation
│   └── video/               # Video generation
├── packages/shared/          # Shared utilities
├── prisma/                  # Database schema
└── infra/                  # Docker configuration
```

### Key Features

#### 🖊️ Document Editor
- Rich text editing with Markdown support
- Version history and comments
- Real-time "Ask AI" on text selection
- Multi-tenant project organization

#### 🤖 AI Chat & RAG
- Contextual chat with document retrieval
- Real-time streaming responses
- Source citations and anchored references
- Multi-format document ingestion (PDF, DOCX, CSV, MD)

#### 🎯 Autonomous Agents
- Multi-step job planning and execution
- Browser automation with Playwright
- Real-time progress streaming
- Job replay functionality
- Usage metering and budget controls

#### 🎨 Media Generation
- AI image generation (Stable Diffusion XL)
- AI video generation (Stable Video Diffusion)
- Artifact management and storage
- Integration with agent workflows

#### 👥 Team Collaboration
- Multi-tenant teams with RBAC
- Project-based organization
- Public sharing capabilities
- Usage quotas and billing

### API Endpoints

#### Web Application
- `/api/auth/*` - Authentication (NextAuth.js)
- `/api/documents/*` - Document CRUD
- `/api/chat` - AI chat with SSE streaming
- `/api/agent/*` - Agent job management
- `/api/artifacts/*` - File and media management

#### Python Services
- `http://localhost:8001/generate` - Image generation
- `http://localhost:8002/generate` - Video generation

### Environment Variables

Key configuration options:

```bash
# Database
DATABASE_URL="postgresql://postgres:password@localhost:5432/manus_clone"

# LLM Provider
PROVIDER="ollama"  # vllm|tgi|ollama|openai|anthropic
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_MODEL="llama3.2:3b"

# Vector Database
VECTOR_BACKEND="qdrant"  # qdrant|pgvector
QDRANT_URL="http://localhost:6333"

# Object Storage
OBJECT_STORE_ENDPOINT="http://localhost:9000"
OBJECT_STORE_BUCKET="manus-artifacts"

# AI Services
IMAGES_API_URL="http://localhost:8001"
VIDEO_API_URL="http://localhost:8002"
```

## 🧪 Testing

```bash
# Unit tests
pnpm test

# E2E tests
pnpm test:e2e
```

## 📦 Deployment

### Docker Production

```bash
# Build all services
docker-compose -f infra/docker-compose.yaml build

# Deploy
docker-compose -f infra/docker-compose.yaml up -d
```

### Environment-Specific Configuration

- **Development**: Local services with hot reload
- **Staging**: Docker containers with external databases
- **Production**: Kubernetes deployment with managed services

## 🔧 Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker logs and ensure ports are available
2. **Model loading errors**: Verify GPU availability and model downloads
3. **Database connection**: Ensure PostgreSQL is running and accessible
4. **Vector search slow**: Check Qdrant configuration and indexing

### Performance Optimization

- Enable GPU acceleration for AI services
- Configure Redis for caching and queues
- Use CDN for static assets
- Implement database connection pooling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- Documentation: [docs/](./docs/)
- Issues: GitHub Issues
- Discussions: GitHub Discussions
