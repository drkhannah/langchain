import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";

dotenv.config();

// Ensure API key is available
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  throw new Error("OPENAI_API_KEY is not set in environment variables");
}

const model = new ChatOpenAI({ 
  model: "gpt-3.5-turbo", 
  openAIApiKey: apiKey
});

const systemTemplate = "Translate the following English text to {language}.";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ['system', systemTemplate],
  ['user', '{text}']
]);

// const messages = [
//   new SystemMessage('Translate the following English text to French.'),
//   new HumanMessage('I love programming.'),
// ];

async function main() {
  try {
    console.log("Invoking model...");
    const promptValue = await promptTemplate.invoke({
      language: 'French',
      text: 'I love programming.'
    });
    const response = await model.invoke(promptValue);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();