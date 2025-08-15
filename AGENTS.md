# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
This will be a fork of CHATWOOT, look on 

# Chatwoot Development Guidelines

## Build / Test / Lint

- **Setup**: `bundle install && pnpm install`
- **Run Dev**: `pnpm dev` or `overmind start -f ./Procfile.dev`
- **Lint JS/Vue**: `pnpm eslint` / `pnpm eslint:fix`
- **Lint Ruby**: `bundle exec rubocop -a`
- **Test JS**: `pnpm test` or `pnpm test:watch`
- **Test Ruby**: `bundle exec rspec spec/path/to/file_spec.rb`
- **Single Test**: `bundle exec rspec spec/path/to/file_spec.rb:LINE_NUMBER`
- **Run Project**: `overmind start -f Procfile.dev`

## Code Style

- **Ruby**: Follow RuboCop rules (150 character max line length)
- **Vue/JS**: Use ESLint (Airbnb base + Vue 3 recommended)
- **Vue Components**: Use PascalCase
- **Events**: Use camelCase
- **I18n**: No bare strings in templates; use i18n
- **Error Handling**: Use custom exceptions (`lib/custom_exceptions/`)
- **Models**: Validate presence/uniqueness, add proper indexes
- **Type Safety**: Use PropTypes in Vue, strong params in Rails
- **Naming**: Use clear, descriptive names with consistent casing
- **Vue API**: Always use Composition API with `<script setup>` at the top

## Styling

- **Tailwind Only**:  
  - Do not write custom CSS  
  - Do not use scoped CSS  
  - Do not use inline styles  
  - Always use Tailwind utility classes  
- **Colors**: Refer to `tailwind.config.js` for color definitions

## General Guidelines

- MVP focus: Least code change, happy-path only
- No unnecessary defensive programming
- Break down complex tasks into small, testable units
- Iterate after confirmation
- Avoid writing specs unless explicitly asked
- Remove dead/unreachable/unused code
- Don’t write multiple versions or backups for the same logic — pick the best approach and implement it
- Don't reference Claude in commit messages

## Project-Specific

- **Translations**:
  - Only update `en.yml` and `en.json`
  - Other languages are handled by the community
  - Backend i18n → `en.yml`, Frontend i18n → `en.json`
- **Frontend**:
  - Use `components-next/` for message bubbles (the rest is being deprecated)

## Ruby Best Practices

- Use compact `module/class` definitions; avoid nested styles

## Architecture Overview

Chatwoot is a full-stack Rails 7.1 application with Vue 3 frontend built using Vite. The architecture supports multi-tenancy through accounts and includes enterprise features.

### Tech Stack

- **Backend**: Ruby 3.4.4, Rails 7.1, PostgreSQL, Redis, Sidekiq
- **Frontend**: Vue 3 (Composition API), Vite 5, Tailwind CSS, Vuex 4
- **Real-time**: ActionCable (WebSockets), Wisper (pub/sub)
- **Background Jobs**: Sidekiq with Redis
- **Testing**: RSpec (Ruby), Vitest (JS/Vue)
- **Deployment**: Docker, Heroku, DigitalOcean supported

### Application Structure

```
app/
├── controllers/           # API and web controllers
│   ├── api/v1/           # REST API endpoints (JSON)
│   ├── super_admin/      # Admin interface controllers  
│   └── webhooks/         # External webhook handlers
├── javascript/           # Frontend Vue.js applications
│   ├── dashboard/        # Main admin dashboard (Vue 3)
│   ├── widget/           # Customer-facing chat widget
│   ├── portal/           # Help center/knowledge base
│   ├── v3/              # New v3 interface components
│   └── shared/           # Shared components across apps
├── models/               # ActiveRecord models and concerns
├── services/             # Business logic layer
├── jobs/                 # Sidekiq background jobs
├── listeners/            # Event handlers (Wisper)
├── policies/             # Authorization (Pundit)
├── channels/             # ActionCable WebSocket channels
└── views/                # Jbuilder JSON templates
```

### Key Models & Relationships

- **Account** → **Users** (multi-tenant structure)
- **Account** → **Inboxes** → **Conversations** → **Messages**
- **Contact** ↔ **ContactInbox** ↔ **Inbox** (many-to-many)
- **Conversation** → **ConversationParticipant** (agents assigned)
- **Channel** types: WebWidget, Email, WhatsApp, Facebook, etc.
- **Enterprise**: Captain AI, SLA policies, custom roles, audit logs

### Communication Channels

Supported via polymorphic `Channel` association on `Inbox`:
- Web Widget (`Channel::WebWidget`)
- Email (`Channel::Email`) 
- WhatsApp (`Channel::Whatsapp`)
- Facebook (`Channel::FacebookPage`)
- Instagram (`Channel::Instagram`)
- Telegram (`Channel::Telegram`)
- Twitter (`Channel::TwitterProfile`)
- SMS/Twilio (`Channel::TwilioSms`)

### Frontend Applications

1. **Dashboard** (`/app/javascript/dashboard/`): Main agent interface
2. **Widget** (`/app/javascript/widget/`): Customer chat widget
3. **Portal** (`/app/javascript/portal/`): Knowledge base frontend
4. **V3** (`/app/javascript/v3/`): New interface components

### Enterprise Features (`enterprise/`)

- **Captain AI**: AI assistant with document indexing and responses
- **SLA Policies**: Service level agreements and tracking
- **Custom Roles**: Role-based permissions beyond default roles
- **Audit Logs**: Comprehensive activity tracking
- **Advanced Reporting**: Enhanced analytics and metrics

### Event System

Uses Wisper pub/sub pattern:
- Models publish events (conversation created, message sent, etc.)
- Listeners handle events (notifications, webhooks, reporting)
- Located in `app/listeners/`

### API Design

- RESTful API at `/api/v1/`
- Account-scoped resources: `/api/v1/accounts/:account_id/...`
- JSON responses using Jbuilder templates
- Authentication via Devise Token Auth
- Authorization via Pundit policies

### Background Processing

- **Sidekiq** for async jobs with Redis
- Job categories: webhooks, notifications, integrations, email processing
- Scheduled via `sidekiq-cron` (see `config/schedule.yml`)

### Database & Search

- **PostgreSQL** primary database with JSONB for flexible schemas
- **pg_search** for full-text search on articles
- **pgvector** + **neighbor** gem for AI embeddings (Captain feature)
- Database triggers via **hairtrigger** for complex logic