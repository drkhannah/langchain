import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

dotenv.config();

const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });

const prompt = PromptTemplate.fromTemplate(
  "Summarize the main themes in these retrieved docs: {context}"
);

const chain = await createStuffDocumentsChain({
  llm: llm,
  outputParser: new StringOutputParser(),
  prompt: prompt,
});

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

const stream = await chain.stream({ context: docs });
for await (const chunk of stream) {
  process.stdout.write(chunk);
}


