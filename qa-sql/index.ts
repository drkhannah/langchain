import "reflect-metadata";
import "dotenv/config";
import { DataSource } from "typeorm";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { SqlDatabase } from "langchain/sql_db";
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { AIMessage, BaseMessage, isAIMessage } from "@langchain/core/messages";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { Document } from "@langchain/core/documents";

const datasource = new DataSource({
  type: "sqlite",
  database: "Chinook.db",
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const vectorStore = new MemoryVectorStore(embeddings);

async function queryAsList(
  database: SqlDatabase,
  query: string
): Promise<string[]> {
  const res: Array<{ [key: string]: string }> = JSON.parse(
    await database.run(query)
  )
    .flat()
    .filter((el: { [key: string]: string } | null) => el != null);
  const justValues: Array<string> = res.map((item) =>
    (Object.values(item)[0] ?? "")
      .replace(/\b\d+\b/g, "")
      .trim()
  );
  return justValues;
}

// Gather entities into a list
let artists: string[] = await queryAsList(db, "SELECT Name FROM Artist");
let albums: string[] = await queryAsList(db, "SELECT Title FROM Album");
let properNouns = artists.concat(albums);

const documents = properNouns.map(
  (text) =>
    new Document({
      pageContent: text,
      metadata: { source: "ChinookDB" },
    })
);

await vectorStore.addDocuments(documents);

const retriever = vectorStore.asRetriever(5);
const retrieverTool = createRetrieverTool(
  retriever,
  {
    name: "searchProperNouns",
    description:
      "Use to look up values to filter on. Input is an approximate spelling " +
      "of the proper noun, output is valid proper nouns. Use the noun most " +
      "similar to the search."
  }
);

const toolkit = new SqlToolkit(db, llm);

const tools = toolkit.getTools();

const systemPromptTemplate = await pull<ChatPromptTemplate>(
  "langchain-ai/sql-agent-system-prompt"
);

const systemMessage = await systemPromptTemplate.format({
  dialect: db.appDataSourceOptions.type,
  top_k: 5,
});

const agent = createReactAgent({
  llm: llm,
  tools: tools,
  prompt: systemMessage,
});

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

let inputs2 = {
  messages: [
    { role: "user", content: "Which country's customers spent the most?" },
  ],
};

for await (const step of await agent.stream(inputs2, {
  streamMode: "values",
})) {
  const lastMessage = step.messages[step.messages.length - 1];
  if (lastMessage) {
    prettyPrint(lastMessage);
  }
  console.log("-----\n");
}

let inputs3 = {
  messages: [{ role: "user", content: "Describe the playlisttrack table" }],
};

for await (const step of await agent.stream(inputs3, {
  streamMode: "values",
})) {
  const lastMessage = step.messages[step.messages.length - 1];
  if (lastMessage) {
    prettyPrint(lastMessage);
  }
  console.log("-----\n");
}

let suffix =
  "If you need to filter on a proper noun like a Name, you must ALWAYS first look up " +
  "the filter value using the 'search_proper_nouns' tool! Do not try to " +
  "guess at the proper name - use this function to find similar ones.";

const system = systemMessage + suffix;

const updatedTools = tools.concat(retrieverTool);

const agent2 = createReactAgent({
  llm: llm,
  tools: updatedTools,
  prompt: system,
});

let inputs4 = {
  messages: [
    { role: "user", content: "How many albums does AC/DC have?" },
  ],
};

for await (const step of await agent2.stream(inputs4, {
  streamMode: "values",
})) {
  const lastMessage = step.messages[step.messages.length - 1];
  if (lastMessage) {
    prettyPrint(lastMessage);
  }
  console.log("-----\n");
}
