---
outline: deep
description: Learn how to write and maintain changelogs for your application using the common-ui.bs.js library.
skillParent: dcc-ui
skillName: changelogs-howto
skillDescription: "End-to-end guide for adding user-facing changelogs to a Nuxt app with @dcc-bs/common-ui.bs.js: the Changelogs component, changelog markdown file format/frontmatter, and version tracking. Use when implementing changelogs or writing release notes (the full how-to companion to the changelogs component reference)."
---

# How to Write Changelogs

This guide explains how to create and maintain changelogs for your application using the `common-ui.bs.js` library.

## Installation

Install the library using your preferred package manager:

```bash
# bun
bun add @dcc-bs/common-ui.bs.js

# npm
npm install @dcc-bs/common-ui.bs.js

# pnpm
pnpm add @dcc-bs/common-ui.bs.js

# yarn
yarn add @dcc-bs/common-ui.bs.js
```

## Overview

The Changelogs component automatically displays new application updates to users in a modal. It tracks which version a user has last seen and only shows newer changelogs on subsequent visits.

**GitHub Repository**: [DCC-BS/common-ui.bs.js](https://github.com/DCC-BS/common-ui.bs.js)

## How It Works

- **First-time visitors**: No modal is shown, but the latest version is stored in localStorage
- **Returning visitors**: If a newer version exists, all newer changelogs are displayed in a modal
- **Version tracking**: The latest viewed version is automatically saved to localStorage
- **No stored version**: Treated as a first visit - nothing is shown

## Setup

### 1. Add the Component to App.vue

Add the `<Changelogs />` component to your main application file or layout for example in the `app/layouts/default.vue`:

```vue
<template>
  <div>
    <Changelogs />
    <!-- Your existing layout content -->
    <slot />
  </div>
</template>
```

### 2. Create the Changelogs Directory

Create a directory for your changelog files:

```bash
mkdir -p server/assets/changelogs
```

## Writing Changelog Files

### File Format

Changelogs are written in Markdown format with frontmatter metadata. Each file represents a version release.

### Frontmatter Structure

Every changelog file must include frontmatter with these fields:

- `title`: Display title for the changelog
- `version`: Semantic version number (e.g., "1.2.0")
- `published_at`: ISO 8601 date format of publication (YYYY-MM-DD)

### Example Changelog

Create a file in `server/assets/changelogs/v1.1.0.md`:

```markdown
---
title: "Version 1.1.0"
version: "1.1.0"
published_at: "2024-01-15"
---

## New Features

- Added new changelog component
- Improved performance
- Enhanced user interface

## Bug Fixes

- Fixed memory leak issue
- Resolved login timeout problem

## Breaking Changes

- Deprecated old API endpoints
```

### Best Practices

**Version Naming**
- Use semantic versioning (MAJOR.MINOR.PATCH)
- File names should match the version (e.g., `v1.2.3.md`)

**Content Organization**
- Use clear section headers (## New Features, ## Bug Fixes, etc.)
- Keep descriptions concise and user-focused
- List items in order of importance

**Common Sections**
- **New Features**: New functionality added
- **Improvements**: Enhancements to existing features
- **Bug Fixes**: Issues that were resolved
- **Breaking Changes**: Changes that require user action
- **Deprecations**: Features being phased out
- **Security**: Security-related updates

### Version Ordering

Changelogs are automatically sorted by version number (newest first). The component fetches all changelogs from `/changelogs` endpoint.

## Testing

For testing purposes, you can set the changelog to an older version by clearing the stored version in localStorage:

```vue
<script setup>
function clearChangelogCache() {
    localStorage.setItem("changelogs-last-read", "0.0.0");
    location.reload();
}
</script>

<template>
  <button @click="clearChangelogCache">
    Clear Changelog Cache
  </button>
  <Changelogs />
</template>
```

## Tips

1. **Write for users, not developers**: Focus on impact rather than technical details
2. **Be consistent**: Use the same format and sections across all changelogs
3. **Publish regularly**: Keep users informed about updates
4. **Group related changes**: Organize items into logical sections
5. **Date your releases**: Always include accurate `published_at` timestamps

## Example Workflow

1. Complete a new feature or release
2. Create a new markdown file in `server/assets/changelogs/`
3. Add frontmatter with version and date
4. Write clear, user-focused descriptions
5. Deploy your application
6. Users will automatically see the new changelog on their next visit
