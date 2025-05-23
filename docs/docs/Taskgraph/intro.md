# Introduction

In complex real-world scenarios, tasks often involve overlapping responsibilities and varied resources. To manage this complexity, workers need mechanisms to identify reusable sub-task patterns while handling unique tasks independently. This capability mirrors real-world problem-solving, where individuals streamline recurring tasks based on past experience while tackling novel challenges flexibly.

By utilizing the frameworks - **TaskGraph**, the agent breaks down intricate processes into manageable tasks, further subdividing them into actionable steps or instructions. The **TaskGraph** serves as a blueprint, providing the guidelines, guardrails, strategies that ensure multiple workers collaborate reliably and efficiently, maintaining control while dynamically adapting to user needs.

## Construction

The **TaskGraph** is a typical graph with **nodes** and **edges**. The nodes represent the steps or the milestones that Agent need to accomplish by utilizing different workers. The content of edge indicates the user's intent and the direction indicate the execution sequence. Consider an e-commerce customer service agent, it needs to manage tasks such as answering inquiries, recommending products, managing orders, processing returns, and more. Each task involves multiple steps, which make use of various resources, such as company policies, product databases, user profiles, or third-party applications. Each step is a node and the resources is hold by the node's attribute. The edge between the steps indicate the execution sequence.

The construction of taskgraph is handled by the **Planner**, which takes config files as input and figures out the tasks scope and the following best practices to complete each task. Finally the **Orchestrator** will handle the navigation of the taskgraph based on the user's query in real-time.
