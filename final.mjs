import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { CohereEmbeddings } from "@langchain/cohere";

import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { ConversationalRetrievalQAChain } from "langchain/chains";
// import { createRetrievalChain } from "langchain/chains/retrieval";
// import { OpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import { BufferMemory } from "langchain/memory";
import { JsonOutputFunctionsParser } from "langchain/output_parsers";

import * as dotenv from "dotenv";

dotenv.config();

const LoadDoc = async () => {
  const mdFile = "documents/aibiliti.md";
  const loader = new TextLoader(mdFile);
  const doc = await loader.load();
  // console.log(doc);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  });

  const splittedDocs = await splitter.splitDocuments(doc);

  const embedding = new CohereEmbeddings();
  console.log("Embedding : ");

  const vectorStore = await HNSWLib.fromDocuments(splittedDocs, embedding);
  console.log("vectorStore : ");

  const vectorStoreRetriever = vectorStore.asRetriever();

  // const model = new ChatAnthropic({
  //    modelNamw: "claude-3-sonnet-20240229",
  // });

  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    
  });
  

  const extractionFunctionSchema = {
    name: "extractor", // Function name (optional)
    description: "Extracts specific data from the response",
    inputs: [], // Input parameters to the function (optional)
    outputs: [
      { name: "data", type: "json" }, // Specify expected output structure
    ],
  };

  const parser = new JsonOutputFunctionsParser({
    schema: extractionFunctionSchema,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStoreRetriever,
    {
      returnSourceDocuments: false,
      memory: new BufferMemory({
        memoryKey: "chat_history",
        inputKey: "question",
        outputKey: "text",
        returnMessages: true,
      }),
      parser
    },
  );
  const company_name = "Aibiliti";

//   const question = `The Document I have provided to you provides the information of a company ${company_name}. I want you to act as a seasoned business analyst. Go through all the information available to you. From this information, extract a company description and its mission in as much detail as possible. `;
// const question = ` The Document I have provided to you provides the information of a company ${company_name}, I want you to go through all the information available to you. Act as a buiseness analyst. Extract names of all products mentioned in the information along with a detailed description for each product. give result in this form  "services":"Array of object and each object contains name and description for example- [{"name":"Name of service","description":"Description of this service"}] }  prompt: I want you to go through all the information available to you. Act as a business analyst. Extract names of all services mentioned in the information along with a detailed description for each service dbKey: "services",note :please remove all "\n" and all extra empty spaces from the extracted text.`;
  // const question = ` The Document I have provided to you provides the information of a company ${company_name}, I want you to go through all the information available to you. Act as a buiseness analyst. Extract names of all products mentioned in the information along with a detailed description for each product. give result in this form  "services":"Array of object and each object contains name and description for example- [{"name":"Name of service","description":"Description of this service"}] }  prompt: I want you to go through all the information available to you. Act as a business analyst. Extract names of all services mentioned in the information along with a detailed description for each service dbKey: "services"`;
  const question = ` The Document I have provided to you provides the information of a company ${company_name}, I want you to go through all the information available to you. Act as a buiseness analyst. Extract names of all services mentioned in the information along with a detailed description for each product. give result in this form  "services":"Array of object and each object contains name and description for example- [{"name":"Name of service","description":"Description of this service"}] }  prompt: I want you to go through all the information available to you. Act as a business analyst. Extract names of all services mentioned in the information along with a detailed description for each service dbKey: "services"`;
  
  const answer = await chain.invoke({ question: question });

  console.log(answer);
  console.log(typeof answer);
 
};
LoadDoc();
