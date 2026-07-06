---
name: dcc-coding
description: "DCC coding standards across languages and stacks. Routes to language-specific conventions for Nuxt/Vue/TypeScript frontends and Python/FastAPI backends. Use when writing or reviewing code in a DCC project and you need naming, structure, tooling, or style conventions for .vue, .ts, or .py files."
---

# dcc-coding

DCC coding standards across languages and stacks. Routes to language-specific conventions for Nuxt/Vue/TypeScript frontends and Python/FastAPI backends. Use when writing or reviewing code in a DCC project and you need naming, structure, tooling, or style conventions for .vue, .ts, or .py files.

# General Coding Standards

Most important thing:

- **Do:** Follow the coding standards for the specific language you are working in.
- **Do:** We want modular code that can be reused across projects.
- **Do:** Use and extend `backend-common` code where possible. See the [backend-common documentation](/backend-common/) for more information.
- **Do:** Use our existing nuxt modules. See the [nuxt documentation](/coding/nuxt) and [user-interface documentation](/user-interface/) for more information.
- **Do:** Add new create new nuxt modules or npm packages for code that is reused across projects.
- **Do:** Use the [Git / GitHub / CI/CD Standards](/dev-setup/git) for your commits and pull requests.
- **Do:** Use the [Dev Setup](/dev-setup) for your development environment.

## Python

- [Python Coding Standards](/coding/python)

## Nuxt / Vue

- [Nuxt / Vue Coding Standards](/coding/nuxt)

## Git / GitHub / CI/CD

- [Git / GitHub / CI/CD Standards](/dev-setup/git)

### Reference Sub-Guidelines
The following reference sub-guides are available in this skill directory. Read them using your file reading tools as needed:

- **[nuxt](references/nuxt.md)**: Nuxt.js + Vue + TypeScript coding conventions: Composition API with <script setup>, file naming (kebab-case pages/PascalCase components/camelCase utils), composables, Nuxt import aliases, Tailwind + Lucide, Biome linting, and Bun dependency management. Use when creating or reviewing .vue/.ts files or Nuxt frontend code.
- **[python](references/python.md)**: Python coding standards and conventions: type hints, Google-style docstrings, FastAPI routers/Pydantic models, Returns for functional error handling, HTTPX async clients, Dependency Injector, and UV/ruff/ty/pytest tooling. Covers code style, structure, and tooling rules (not the shared backend-common library modules). Use when writing or reviewing .py backend code.
