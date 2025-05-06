# Generative AI

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Generative AI](#generative-ai)
- [AI](#ai)
- [Machine learning](#machine-learning)
   * [How does machine learning work?](#how-does-machine-learning-work)
- [Deep learning](#deep-learning)
- [Conventional AI systems](#conventional-ai-systems)
- [GenAI systems](#genai-systems)
- [ChatGPT](#chatgpt)
   * [Large Language Models](#large-language-models)
- [Prompts](#prompts)
   * [Prompt engineering](#prompt-engineering)
   * [Best practices for prompt engineering ](#best-practices-for-prompt-engineering)
- [Embeddings](#embeddings)
- [Fine tuning](#fine-tuning)
   * [Fine tuning limitations](#fine-tuning-limitations)
- [Summary of concepts](#summary-of-concepts)
- [GenAI use cases](#genai-use-cases)
   * [In the software industry](#in-the-software-industry)
- [Gen AI workflow for creating a chatbot](#gen-ai-workflow-for-creating-a-chatbot)
   * [Chatbot code example](#chatbot-code-example)

<!-- TOC end -->

[Generative AI for beginners Udemy course](https://www.udemy.com/course/generative-ai-for-beginners-b/?srsltid=AfmBOor5eMWBtKCoD5S6PY7MeatyG_l9ho9b7YyESOYpPD2q8sxmidEv&couponCode=LEARNNOWPLANS)

<!-- TOC --><a name="generative-ai"></a>
## Generative AI

`Generative AI = Generative + AI`

* Generative: Generate content (Text, image, audio, ...)
* AI: Using artificial intelligence

<!-- TOC --><a name="ai"></a>
## AI

Broad computer science field that develops intelligent systems capable of performing tasks typically requiring human intelligence

<!-- TOC --><a name="machine-learning"></a>
## Machine learning

Subset of AI that develops **algorithms** and **models** that enable computers to learn and *make predictions* or *decisions* without explicit programming.

<!-- TOC --><a name="how-does-machine-learning-work"></a>
### How does machine learning work?

We memorize objects as humans and can guess it the next time it is shown to us. For example, a child learns what an apple is by looking at a picture or drawing. Later, when shown an apple, it can identify it.

Computers too need to memorize. However, they need *lots and lots of data* to be able to identify objects based on the training. This is called **training data** and is typially in the millions or billions.

Processing massive training data requires **computational power** which did not exist earlier but now thanks to advanced GPUs, we can process and optimize large amounts of data.

Machines also need **algorithms** that will use the data and take decisions without human intervention.

`Machine learning = Training data + Computational power + Algorithms`

<!-- TOC --><a name="deep-learning"></a>
## Deep learning

It is a subset of Machine Learning that focuses on teaching computers to learn and make decisions by processing data through **neural networks** inspired by the human brain.

Neural networks can be thought of as stages or layers. Each layer does a bit of processing, learns from it, and passes on the information to the next layer. *More the layers, the more accurate the processing*. 

It was computationally expensive to build and adopt Neural Networks but advancements in GPU make it possible today and many GenAI tools like ChatGPT use it.

- Machine Learning is a subset of AI
- Deep learning is a subset of Machine Learning
- **Generative AI is a subset of Deep Learning**

<!-- TOC --><a name="conventional-ai-systems"></a>
## Conventional AI systems

Conventional AI systems took in input data (such as training data and new data) and made one of the following:
1. Prediction
2. Classification
3. Clustering
4. NLP
5. Computer Vision

*It did not generate anything new!*

For example, a conventional AI trained on millions of images of apples can tell us whether a given image is of an apple or not!

<!-- TOC --><a name="genai-systems"></a>
## GenAI systems

These take in training data (much more than conventional AIs) and generate new content. This content is not extracted from the input data but is instead a newly created one! 

For example, a GenAI trained on millions of images of apples can produce an image of an apple on command and this apple was not part of any input data set

GenAI has shifted software products to adopt a more ***conversational or contextual understanding*** style.

*Non-AI systems are stateless but the way humans remember things is stateful so maintaining the context helps GenAI products.*

- **Conversational**: No one wants to google a question and see 15 links but instead they want to get an answer their question straight away.
- **Contextual understanding**: If I ask a question about delhi and later ask for "what the weather is like there", it should know that I am asking about Delhi's weather.

One such GenAI product is ChatGPT!

<!-- TOC --><a name="chatgpt"></a>
## ChatGPT

- It is a GenAI product 
- Has a conversational and contextual understanding style: ***It excels at generating human like responses (A game changer)***
- Built by OpenAI and integrated into Microsoft products like Azure and VScode
- Based on the **Generative Pre-Trained (GPT)** architecture, a type of *neural network*
- Trained on billions of documents (text, video, images, etc). For example, ChatGPT 3.0 was trained on 500GB+ textual data.
- Has a web interface for end users as well as an API for developers
- It is a **Large Language Model (LLM)**

It is **not designed for tackling complex mathematical problems** but can perform things like translation, generating text responses, and summarizing documents.

<!-- TOC --><a name="large-language-models"></a>
### Large Language Models

Large Language Models (**LLMs**) are **powerful models designed for understanding and generating human-like text.**

LLM means **"text"**. It can understand grammar, sentences, sentiment, etc contained in text. However, **`GenAI != LLM`** since GenAI can be any media such as text, images, videos, and so on.

LLMS use a type of *neural network* called **Transformers**. These transformers are great at understanding the meaning and context of text. The LLMs are trained on millions of textual data and **provide one output at a time i.e one word at a time**. Every time you type into ChatGPT and it is generating a long paragraph word by word, it is because it is outputting one word only at a time and trying to generate the next word later, and so on. (Bit by bit)

**Key points on LLMs:**
* LLMs are **pre-trained**: Huge corpus of data
* **LLMs use massive neural networks**. Neural networks have **parameters** and these are like variables that tune the network to improve its accuracy. LLMs like ChatGPT and Google Palm are trained on *100s of billions* of parameters
* LLMs can be **fine-tuned**: They come pre-trained but can be provided additional data sets to tweak the responses to be more accurate w.r.t that domain. For example, using an LLM but fine tuning it for healthcare can be achieved by using an additional healthcare dataset.

**Use cases for LLMs:**
* Content generation - Marketing, advertising, ...
* Chatbots and virtual assistants - User support, interactions, ...
* Language translation - expand communication
* Text summarization - reduce lengthy content
* QnA - ask questions, provide information

<!-- TOC --><a name="prompts"></a>
## Prompts

A prompt is a specific question, command, or input you provide to an AI system to request a particular response, information, or action.

For example, "Write a python program to add two numbers" is a statement which is a prompt to the LLM.

<!-- TOC --><a name="prompt-engineering"></a>
### Prompt engineering

Providing prompts to LLMs can sometimes be broad. In order to get specific responses from it, we need to ask it pin pointed questions. This is known as prompt engineering.

For example, a prompt such as "Tell me about solar energy" is too broad and will elicit a broad answer. However, "What are the recent developments in solar energy, especially in the timeframe between 2020 and 2025?" is a very specific prompt and we will receive a specific answer too!

<!-- TOC --><a name="best-practices-for-prompt-engineering"></a>
### Best practices for prompt engineering 
 
 * **Clearly convey the desired response**: Describe what type of data you want
 * **Provide context or background information**: For example, if you want the answer to be in the tone of a journalist then writing something such as "Imagine you are a journalist...", i.e setting the role, works
 * **Balance simplicity and complexity**: If you provide too many cues, the LLM may get confused or provide a very verbose answer. Determine the how simple or complex the answer needs to be
 * **Iterative testing and refinement**: Keep working on your prompts and get an understanding of what works and what doesn't

**Cons of prompt engineering:** It is difficult to get consistent answers when the prompts change even slightly (or if the model changes or updates)
 
<!-- TOC --><a name="embeddings"></a>
## Embeddings

Machines understand only numbers, so how can they understand text (fundamental question)? The answer is that they don't!

Each word entered is converted to a list of numbers by the neural network used by the LLM. These numbers are known as an **embeddings**.

How is this number generated? It is quite complex. The LLM is trained on billions or trillions of data sets and it is able to associate numbers based on how closely associated two words are. *This is what is uses to "generate" the next word!* 

For example,  "I ate ice cream" might have the embeddings:
"I": `0.412424,  0.2432, ...`
"ate": `0.752994,  0.59110, ...`
"ice": `0.474743,  0.37112, ...`
"cream": `0.472844,  0.3955, ...`

**Note:**
1. Generation cue: Notice how the words "ice" and "cream" embeddings are quite similar in this example. This is because "ice cream" is a common phrase and the LLM must have been trained on billions of data sets where they occur together.
2. Meaning: "I just had a great ice cream" versus "My ice cream melted, great!" involves two different emotions, happiness and sarcasm, respectively. The LLMs are able to generate embeddings based on the meanings too! 

**Embeddings are used to understand text semantics, meanings, context, etc from trillions of training datasets!**. Embeddings transform data into **vectors** that capture meaningful relationships, helping AI better understand and work with the information.

Therefore, for a text, "I am going to the mall to have ice", the LLM may be able to guess "cream" as the next word! 

* In the context of AI, a vector is an array of numbers that represents data. 
* Each number in the vector corresponds to a specific feature of the data. For example, in natural language processing, words or phrases can be represented as vectors where each dimension captures some aspect of the word's meaning. 
* This allows AI models to work with complex data in a structured and quantifiable way, making it easier to perform tasks like classification, clustering, and other forms of analysis.

<!-- TOC --><a name="fine-tuning"></a>
## Fine tuning

Fine tuning is adapting a pre-trained model to perform specific tasks or to cater to a particular domain more effectively.

There are 3 types of fine-tuning:
1. **Self-supervised**: Provide the domain specific training data and let the LLM learn on its own. (Ex: provide healthcare documents to the LLM)
2. **Supervised**: Provide input and expected outputs to the LLM to train based on that (Ex: Input="How can I check for broken bones?" Output="x-ray" is fed to the LLM)
3. **Reinforcement**: An old method of rewarding the model for a successful guess which it uses as feedback to improve its subsequent guesses.

<!-- TOC --><a name="fine-tuning-limitations"></a>
### Fine tuning limitations

1. Does not create intelligence from scratch
2. Data requirement is a must - Without training on a given data set, LLMs cannot fine tune
3. It is not a single universal solution
4. It is not a magical one time process - It requires iterations to perfect

<!-- TOC --><a name="summary-of-concepts"></a>
## Summary of concepts

```
[USER]

||
||
\/

[PROMPT/QUERY]   <------ PROMPT ENGINEERING

||
||
\/

[LLMS: 
NEURAL NETWORKS (TRANSFORMER), 
EMBEDDINGS, 
TRAINING DATA]   <------ FINE TUNED BY PROVIDING DOMAIN SPECIFIC DATA

||
||
\/

[OUTPUT]
```

<!-- TOC --><a name="genai-use-cases"></a>
## GenAI use cases

<!-- TOC --><a name="in-the-software-industry"></a>
### In the software industry

| Use case | Details | Benefits |
|--|--|--|
|  Build | Generate code from scratch, fix errors, optimize performance, Direct integration with IDEs | Reduce manual effort, cross-enable developers not pro at another programming language, better code practices, documented details |
| Testing | Identify test scenarios, write detailed test cases, generate automation scripts | Reduce scenario misses, identify edge cases, reduce manual effort, expedite testing cycle |
| Requirement gathering | Generate epics, user stories, and acceptance criteria | Ensure coverage and identify edge cases, reduce manual effort, expedite the cycle |
| Documentation | Generate documentation: requirement docs, test reports, user guides / operational docs | Reduce manual effort, ensure proper documentation, more regulatory and organizational needs |

<!-- TOC --><a name="gen-ai-workflow-for-creating-a-chatbot"></a>
## Gen AI workflow for creating a chatbot

When you need to feed data to an LLM and make it use that to answer questions.

**PROVIDING CUSTOM DATA TO AN LLM**
```
**[SOURCE]** (Example: PDF file of the Constitution of India)
|
|
-> **[CHUNKS]** (Example: Few pages of the PDF at a time. This is for better ingestion)
|
|
-> **[EMBEDDINGS]** (Example: Converts the chunks to numbers called embeddings - Uses an LLM provider lib (OpenAI))
|
|
-> **[VECTOR STORE]** (Example: It is a database storing an embedding value mapped to its chunk)
```

**QUERYING**
```
**[USER]**
|
|
-> **[WRITES A QUERY]**
|
|
-> **[SEMANTIC SEARCH]** --> (OCCURS ON) --> **[VECTOR STORE]** 
|
|
-> (RANKED RESULTS I.E MATCHING CHUNKS SENT TO) --> **[LLM]** (Ex: ChatGPT) --> Result back to user
```

<!-- TOC --><a name="chatbot-code-example"></a>
### Chatbot code example

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "<OPENAI_API_KEY>" #Pass your key here

#Upload PDF files
st.header("My first Chatbot")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)


    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
```
