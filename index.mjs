import { ChatAnthropic } from "@langchain/anthropic";
import { config } from 'dotenv';
config();

// Now you can use process.env.ANTHROPIC_API_KEY
const model = new ChatAnthropic({
  temperature: 0.9,
  modelName: "claude-3-sonnet-20240229",
  // In Node.js defaults to process.env.ANTHROPIC_API_KEY,
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  maxTokens: 1024,
});

async function getResponse() {
  const res = await model.invoke("name of the 3rd planet from the sun?");
  console.log(res);
}

getResponse();