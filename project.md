This is a proof-of-concept project to learn more about MCP and agentic workflows.

## Goal

An automated machine learning system that, given a dataset and a prediction target, can develop and run scripts to train and evaluate machine learning models to predict that target.

There are two primary concerns: 
1. Code for working with the file system/reading and summarizing the data/writing and running files.
2. Code for managing an agent's state through the process of exploring, building, and evaluating models.

To maintain a clean separation of these concerns, there should be two entirely separate pieces to this project. For 1., an MCP server that surfaces basic filesystem operations and other necessary tools. For 2., an LLM-powered flow that uses LangGraph.

## Assumptions

While we eventually want more flexibility, in the short term, we assume that the input dataset is a directory containing two files: a dataset in csv form, and a text file describing what the prediction goal is (target variable, preferred evaluation metrics, etc).

## Details

For the MCP server, any filesystem access it provides should be relative to the server's working directory, so that the LLM can't get arbitrary access to the broader filesystem.

To provide visibility into the machine learnining process, the training and eval scripts produced by the agent should use mlflow. Other packages that should be available to it: seaborn, matplotlib, scikit-learn, polars