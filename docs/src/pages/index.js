import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import arklexDiagram from "@site/static/img/AgentOrg-main.png";

import Heading from "@theme/Heading";
import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={styles.button}
            href="https://pypi.org/project/arklex/"
          >
            <svg
              height="24"
              preserveAspectRatio="xMidYMid"
              viewBox="0 0 256 255"
              width="24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <linearGradient
                id="a"
                x1="12.959359%"
                x2="79.638833%"
                y1="12.039393%"
                y2="78.200854%"
              >
                <stop offset="0" stop-color="#387eb8" />
                <stop offset="1" stop-color="#366994" />
              </linearGradient>
              <linearGradient
                id="b"
                x1="19.127525%"
                x2="90.741533%"
                y1="20.579181%"
                y2="88.429037%"
              >
                <stop offset="0" stop-color="#ffe052" />
                <stop offset="1" stop-color="#ffc331" />
              </linearGradient>
              <path
                d="m126.915866.07227555c-64.8322829.00000462-60.7837372 28.11518925-60.7837372 28.11518925l.0722755 29.1270467h61.8678717v8.7453417h-86.4415589s-41.486166-4.7049094-41.486166 60.7114618c-.00000463 65.416358 36.2100508 63.096556 36.2100508 63.096556h21.6103896v-30.355731s-1.1648552-36.210051 35.6318464-36.210051h61.3619421s34.475438.557297 34.475438-33.3190286v-56.0135516c0-.0000047 5.234323-33.89723325-62.518352-33.89723325zm-34.1140591 19.58667415c6.1553999-.0000045 11.1304351 4.9750349 11.1304351 11.1304348.000004 6.1553999-4.9750352 11.1304348-11.1304351 11.1304348-6.1553999.0000046-11.1304348-4.9750349-11.1304348-11.1304348-.0000047-6.1553999 4.9750349-11.1304348 11.1304348-11.1304348z"
                fill="url(#a)"
              />
              <path
                d="m128.757101 254.126271c64.832302 0 60.783738-28.11519 60.783738-28.11519l-.072275-29.127046h-61.867872v-8.745342h86.441559s41.486166 4.704896 41.486166-60.711485c.000023-65.4163514-36.210051-63.0965491-36.210051-63.0965491h-21.61039v30.3557243s1.164874 36.2100508-35.631846 36.2100508h-61.361948s-34.475437-.557296-34.475437 33.319052v56.013552s-5.2343225 33.897233 62.518356 33.897233zm34.114059-19.586674c-6.155401 0-11.130434-4.975033-11.130434-11.130435 0-6.155403 4.975033-11.130435 11.130434-11.130435 6.155403 0 11.130435 4.975032 11.130435 11.130435.000023 6.155402-4.975032 11.130435-11.130435 11.130435z"
                fill="url(#b)"
              />
            </svg>
            <span className={styles.pythonButtonText}>pip install arklex</span>
          </Link>
          <Link className={styles.button} to="/docs/intro">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path
                d="M6 6C6 5.44772 6.44772 5 7 5H17C17.5523 5 18 5.44772 18 6V20L12 17L6 20V6Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Documentation
          </Link>
          <Link
            className={styles.button}
            href="https://github.com/arklexai/Agent-First-Organization"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            Source Code
          </Link>
          <Link className={styles.button} href="https://discord.gg/jpZs3GqnTG">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.118.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
            </svg>
            Get Connected
          </Link>
        </div>
      </div>
    </header>
  );
}

function MainContent() {
  return (
    <section className={styles.contentSection}>
      <div className={styles.mainContent}>
        <div className={styles.mediaSection}>
          <div className={styles.imageWrapper}>
            <img src={arklexDiagram} alt="Arklex Diagram" />
          </div>
          <div className={styles.videoWrapper}>
            <div className={styles.videoFrame}>
              <iframe
                src="https://www.loom.com/embed/f5a45bcc8c834ec998083eaf7793a912?sid=7525fc12-ffd2-403c-9623-cab087bb2ced"
                title="Arklex: The Future of AI Agents"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                webkitallowfullscreen="true"
                mozallowfullscreen="true"
              />
            </div>
          </div>
        </div>
        <div className={styles.ctaContainer}>
          <div className={styles.ctaLine} />
          <Link
            className={styles.ctaButton}
            href="https://www.arklex.ai/qa/blogs/57526a0e-7803-4452-96f3-53995394accd"
          >
            Learn more about Arklex
            <svg
              viewBox="0 0 24 24"
              fill="none"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />"
    >
      <HomepageHeader />
      <MainContent />
      <HomepageFeatures />
    </Layout>
  );
}
