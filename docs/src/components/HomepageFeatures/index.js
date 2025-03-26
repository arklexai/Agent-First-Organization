import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "Mixed Control",
    Svg: require("@site/static/img/AgentOrg-1.svg").default,
    description: (
      <>
        Enables agents to address diverse goals driven by user needs and builder
        objectives, fostering dynamic and goal-oriented interactions.
      </>
    ),
    position: "top-left",
  },
  {
    title: "Task Composition",
    Svg: require("@site/static/img/AgentOrg-2.svg").default,
    description: (
      <>
        Breaks down complex real-world tasks into modular, reusable components
        managed by individual workers, promoting efficiency and scalability.
      </>
    ),
    position: "top-right",
  },
  {
    title: "Human Intervention",
    Svg: require("@site/static/img/AgentOrg-3.svg").default,
    description: (
      <>
        Integrates human oversight and enables interactive refinement, ensuring
        critical decisions are accurate and user preferences are prioritized.
      </>
    ),
    position: "bottom-left",
  },
  {
    title: "Continual Learning",
    Svg: require("@site/static/img/AgentOrg-4.svg").default,
    description: (
      <>
        Allows agents to evolve and improve by learning from interactions,
        ensuring sustained relevance and effectiveness in dynamic environments.
      </>
    ),
    position: "bottom-right",
  },
];

function Feature({ Svg, title, description, position }) {
  return (
    <div className={clsx(styles.featureCard, styles[position])}>
      <div className={styles.svgWrapper}>
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className={styles.textContent}>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <div>
      <div className={styles.sectionTitle}>
        <h2>Core Features</h2>
      </div>
      <section className={styles.features}>
        <div className={styles.gridContainer}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </section>
    </div>
  );
}
