import { ChatAnthropic } from "@langchain/anthropic";
import { config } from 'dotenv';
config();

// Now you can use process.env.ANTHROPIC_API_KEY
const model = new ChatAnthropic({
  temperature: 0.9,
  modelName: "claude-3-sonnet-20240229",
  maxTokens: 1024,
});

async function getResponse() {
  const res = await model.invoke("distance between earth and moon?");
  console.log(res);
}

getResponse();