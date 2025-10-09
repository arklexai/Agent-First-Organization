import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Mixed Control',
    Svg: require('@site/static/img/AgentOrg-1.svg').default,
    description: (
      <>
        Enables agents to address diverse goals driven by user needs and builder objectives.
      </>
    ),
  },
  {
    title: 'Task Composition',
    Svg: require('@site/static/img/AgentOrg-2.svg').default,
    description: (
      <>
        Breaks down complex tasks into modular, reusable components for efficiency and scalability.
      </>
    ),
  },
  {
    title: 'Human Intervention',
    Svg: require('@site/static/img/AgentOrg-3.svg').default,
    description: (
      <>
        Integrates human oversight and interactive refinement for accurate decisions.
      </>
    ),
  },
  {
    title: 'Continual Learning',
    Svg: require('@site/static/img/AgentOrg-4.svg').default,
    description: (
      <>
        Allows agents to evolve and improve through interaction learning.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.svgWrapper}>
        <Svg className={styles.featureSvg} role='img' />
      </div>
      <div className={styles.textContent}>
        <Heading as='h3'>{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <div className={styles.container}>
      <div className={styles.sectionTitle}>
        <h2>Core Features</h2>
      </div>
      <section className={styles.features}>
        <div className={styles.featuresGrid}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </section>
    </div>
  );
}
