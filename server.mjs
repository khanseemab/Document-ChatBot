import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { CohereEmbeddings } from "@langchain/cohere";

import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { ConversationalRetrievalQAChain } from "langchain/chains";
// import { createRetrievalChain } from "langchain/chains/retrieval";
// import { OpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { BufferMemory } from "langchain/memory";

import * as dotenv from "dotenv";

dotenv.config();
// PDF Load
const getDocAnswer = async () => {
  const loader = new PDFLoader("documents/budget_speech.pdf");
  // const pdf = new PDFLoader("documents/giver.pdf");

  console.log("Loading Document :");
  const docs = await loader.load();


  // Splitter Function
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  });

  // Chunks Created from PDF
  const splittedDocs = await splitter.splitDocuments(docs);

  // console.log(splittedDocs);

  // Embeddings

  // OPEN AI EMBEDDING
  console.log("EMBEDDING : ");

  const embedding = new CohereEmbeddings();

  // Vector Store

  console.log("vectorStore : ");
  const vectorStore = await HNSWLib.fromDocuments(splittedDocs, embedding);

  //Retriever

  const vectorStoreRetriever = vectorStore.asRetriever();

  console.log("Initializing ChatAnthropic : ");
  const model = new ChatAnthropic({
    modelName: "claude-3-sonnet-20240229",
  });

  // Retrieval QA Chain

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStoreRetriever,
    {
      returnSourceDocuments: false,
      memory: new BufferMemory({
        memoryKey: "chat_history",
        inputKey: "question", // The key for the input to the chain
        outputKey: "text", // The key for the final conversational output of the chain
        returnMessages: true, // If using with a chat model (e.g. gpt-3.5 or gpt-4)
      }),
    }
  );
  //   console.log("ConversationalRetrievalQAChain : ");

  //   const chain =  createRetrievalChain.fromLLM({
  //     model,
  //     vectorStoreRetriever,
  //   });

  // const chain =  MultiRetrievalQAChain.fromLLMAndRetrievers({
  //     model,
  //     vectorStoreRetriever,

  //   });
  // Question

  const question = "What is the Theme of g20";

  const answer = await chain.invoke({ question: question });
  console.log("Answer : ", answer);
//     const followUpRes = await chain.invoke({
//       question: "Was that nice?",
//     });
//     console.log(followUpRes);
};

getDocAnswer();
