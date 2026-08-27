import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import "cheerio";
import { pull } from "langchain/hub";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { TokenTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { Annotation, Send, StateGraph } from "@langchain/langgraph";
import { collapseDocs, splitListOfDocs } from "langchain/chains/combine_documents/reduce";

dotenv.config();

const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

const mapPrompt = await pull<ChatPromptTemplate>("rlm/map-prompt");

const reducePrompt = await pull<ChatPromptTemplate>("rlm/reduce-prompt");

const textSplitter = new TokenTextSplitter({ chunkSize: 1000, chunkOverlap: 0 });
const splitDocs = await textSplitter.splitDocuments(docs);
console.log(`Split into ${splitDocs.length} docs.`);

let tokenMax = 1000;

async function lengthFunction(documents) {
  const tokenCounts = await Promise.all(
    documents.map(async (doc) => {
      return llm.getNumTokens(doc.pageContent);
    })
  );
  return tokenCounts.reduce((sum, count) => sum + count, 0);
}

const OverallState = Annotation.Root({
  contents: Annotation<string[]>,
  summaries: Annotation<string[]>({
    reducer: (state, update) => state.concat(update),
  }),
  collapsedSummaries: Annotation<Document[]>,
  finalSummary: Annotation<string>,
});

interface SummaryState {
  content: string;
}

// Here we generate a summary, given a document
const generateSummary = async (
  state: SummaryState
): Promise<{ summaries: string[] }> => {
  const prompt = await mapPrompt.invoke({ docs: state.content });
  const response = await llm.invoke(prompt);
  return { summaries: [String(response.content)] };
};

// Here we define the logic to map out over the documents
// We will use this as an edge in the graph
const mapSummaries = (state: typeof OverallState.State) => {
  // We will return a list of `Send` objects
  // Each `Send` object consists of the name of a node in the graph
  // as well as the state to send to that node
  return state.contents.map(
    (content) => new Send("generateSummary", { content })
  );
};

const collectSummaries = async (state: typeof OverallState.State) => {
  return {
    collapsedSummaries: state.summaries.map(
      (summary) => new Document({ pageContent: summary })
    ),
  };
};

async function _reduce(input: Document[]) {
  // Format the documents for the prompt template
  const formattedDocs = input.map(doc => doc.pageContent).join("\n\n");
  const prompt = await reducePrompt.invoke({ doc_summaries: formattedDocs });
  const response = await llm.invoke(prompt);
  return String(response.content);
}

// Add node to collapse summaries
const collapseSummaries = async (state: typeof OverallState.State) => {
  const docLists = splitListOfDocs(
    state.collapsedSummaries,
    lengthFunction,
    tokenMax
  );
  const results = [];
  for (const docList of docLists) {
    results.push(await collapseDocs(docList, _reduce));
  }

  return { collapsedSummaries: results };
};

// This represents a conditional edge in the graph that determines
// if we should collapse the summaries or not
async function shouldCollapse(state: typeof OverallState.State) {
  let numTokens = await lengthFunction(state.collapsedSummaries);
  if (numTokens > tokenMax) {
    return "collapseSummaries";
  } else {
    return "generateFinalSummary";
  }
}

// Here we will generate the final summary
const generateFinalSummary = async (state: typeof OverallState.State) => {
  const response = await _reduce(state.collapsedSummaries);
  return { finalSummary: response };
};

const graph = new StateGraph(OverallState)
  .addNode("generateSummary", generateSummary)
  .addNode("collectSummaries", collectSummaries)
  .addNode("collapseSummaries", collapseSummaries)
  .addNode("generateFinalSummary", generateFinalSummary)
  .addConditionalEdges("__start__", mapSummaries, ["generateSummary"])
  .addEdge("generateSummary", "collectSummaries")
  .addConditionalEdges("collectSummaries", shouldCollapse, [
    "collapseSummaries",
    "generateFinalSummary",
  ])
  .addConditionalEdges("collapseSummaries", shouldCollapse, [
    "collapseSummaries",
    "generateFinalSummary",
  ])
  .addEdge("generateFinalSummary", "__end__");

const app = graph.compile();

// Save graph visualization to file
const image = await app.getGraph().drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();
const fs = await import("fs");
fs.writeFileSync("graph.png", new Uint8Array(arrayBuffer));
console.log("Graph saved as graph.png");

let finalSummary = null;

for await (const step of await app.stream(
  { contents: splitDocs.map((doc) => doc.pageContent) },
  { recursionLimit: 10 }
)) {
  console.log(Object.keys(step));
  if (step.hasOwnProperty("generateFinalSummary")) {
    finalSummary = step.generateFinalSummary;
    console.log("\n\nFinal Summary:\n");
    console.log(finalSummary);
  }
}