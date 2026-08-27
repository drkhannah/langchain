import "reflect-metadata";
import "dotenv/config";
import { Annotation, MemorySaver, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { QuerySqlTool } from "langchain/tools/sql";

const datasource = new DataSource({
  type: "sqlite",
  database: "Chinook.db",
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  query: Annotation<string>,
  result: Annotation<string>,
  answer: Annotation<string>,
});

const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });

// Define the SQL query prompt template directly instead of pulling from hub
const queryPromptTemplate = await pull<ChatPromptTemplate>(
  "langchain-ai/sql-query-system-prompt"
);

queryPromptTemplate.promptMessages.forEach((message) => {
  console.log(message.lc_kwargs.prompt.template);
});

const queryOutput = z.object({
  query: z.string().describe("Syntactically valid SQL query."),
});

const structuredLlm = llm.withStructuredOutput(queryOutput);

const writeQuery = async (state: typeof InputStateAnnotation.State) => {
  const promptValue = await queryPromptTemplate.invoke({
    dialect: db.appDataSourceOptions.type,
    top_k: 10,
    table_info: await db.getTableInfo(),
    input: state.question,
  });
  const result = await structuredLlm.invoke(promptValue);
  return { query: result.query };
};

const executeQuery = async (state: typeof StateAnnotation.State) => {
  const executeQueryTool = new QuerySqlTool(db);
  return { result: await executeQueryTool.invoke(state.query) };
};

const generateAnswer = async (state: typeof StateAnnotation.State) => {
  const promptValue =
    "Given the following user question, corresponding SQL query, " +
    "and SQL result, answer the user question.\n\n" +
    `Question: ${state.question}\n` +
    `SQL Query: ${state.query}\n` +
    `SQL Result: ${state.result}\n`;
  const response = await llm.invoke(promptValue);
  return { answer: response };
};

const graphBuilder = new StateGraph({
  stateSchema: StateAnnotation,
})
  .addNode("writeQuery", writeQuery)
  .addNode("executeQuery", executeQuery)
  .addNode("generateAnswer", generateAnswer)
  .addEdge("__start__", "writeQuery")
  .addEdge("writeQuery", "executeQuery")
  .addEdge("executeQuery", "generateAnswer")
  .addEdge("generateAnswer", "__end__");

const checkpointer = new MemorySaver();

const graphWithInterrupt = graphBuilder.compile({
  checkpointer,
  interruptBefore: ["executeQuery"],
});

// Now that we're using persistence, we need to specify a thread ID
// so that we can continue the run after review.
const threadConfig = {
  configurable: { thread_id: "1" },
  streamMode: "updates" as const,
};

// Save graph visualization to file
const image = await graphWithInterrupt.getGraph().drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();
const fs = await import("fs");
fs.writeFileSync("graph.png", new Uint8Array(arrayBuffer));
console.log("Graph saved as graph.png");

let inputs = { question: "How many employees are there?" };

console.log(inputs);
console.log("\n====\n");
for await (const step of await graphWithInterrupt.stream(
  inputs,
  threadConfig
)) {
  console.log(step);
  console.log("\n====\n");
}

for await (const step of await graphWithInterrupt.stream(null, threadConfig)) {
  console.log(step);
  console.log("\n====\n");
}
