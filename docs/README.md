# Arklex AI Documentation

This directory contains the complete documentation for Arklex AI, built with [Docusaurus](https://docusaurus.io/).

## 📖 Documentation Access

- **🌐 Live Documentation**: [https://arklexai.github.io/Agent-First-Organization/](https://arklexai.github.io/Agent-First-Organization/)
- **📁 Local Files**: Browse the markdown files directly in this directory
- **🚀 Local Development**: Follow the setup instructions below

## 🚀 Local Development

### Prerequisites

- [Node.js](https://nodejs.org/en/download/) version 18.0 or above
- npm version 9.0 or above

Installation through nvm:

```bash
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

# download and install Node.js (you may need to restart the terminal)
nvm install 22

# verifies the right Node.js version is in the environment
node -v # should print `v22.11.0`

# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
```

### Setup & Run

```bash
# Navigate to docs directory
cd docs

# Install dependencies
npm install

# Start development server
npm run start
```

The site will be available at [http://localhost:3000/](http://localhost:3000/).

### Production Build

```bash
# Test production build locally
npm run serve
```

## 📁 Documentation Structure

- **`docs/docs/`** - Main documentation pages
  - **`Config/`** - Configuration guides
  - **`Example/`** - Usage examples and tutorials
  - **`Integration/`** - Third-party integrations
  - **`Workers/`** - Worker documentation
  - **`Evaluation/`** - Testing and evaluation guides
- **`docs/static/`** - Static assets (images, etc.)
- **`docs/src/`** - Docusaurus source files

### 🚀 Deployment

```bash
# Set your GitHub username
export GIT_USER=<your-github-user-name>

# Deploy to GitHub Pages
npm run deploy
```

## 📝 Contributing

To contribute to the documentation:

1. Edit the markdown files in `docs/docs/`
2. Test locally with `npm run start`
3. Submit a pull request

For more information, see the [Contributing Guide](../../CONTRIBUTING.md).
