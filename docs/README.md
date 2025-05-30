# How to host the documentation locally

## Pre-requisites

- [Node.js](https://nodejs.org/en/download/) version 18.0 or above:
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

## Install Docusaurus required dependencies

  ```bash
  cd docs # navigate to the docs folder from the root directory: AgentOrg/docs
  npm install
  ```

## Start the site

  ```bash
  npm run start
  ```

The `npm run start` command builds your website locally and serves it through a development server, ready for you to view at <http://localhost:3000/>.
  
Test your production build locally:

  ```bash
  npm run serve
  ```
The `build` folder is now served at [http://localhost:3000/](http://localhost:3000/).


## Deploy your site

Docusaurus is a **static-site-generator** (also called **[Jamstack](https://jamstack.org/)**).

It builds your site as simple **static HTML, JavaScript and CSS files**.

Build your site **for production**:

1. Make sure you're in the `docs` directory
2. Set your GitHub username as an environment variable:
   ```bash
   export GIT_USER=<your-github-user-name>
   ```
3. Run the deployment command:
   ```bash
   npm run deploy
   ```
The static files are generated in the `build` folder.
