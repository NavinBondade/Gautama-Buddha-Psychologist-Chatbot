# Gautama-Buddha-Psychologist-Chatbot
<p align="center">
<img src="https://e0.pxfuel.com/wallpapers/344/189/desktop-wallpaper-buddha-anime-buddhist-art.jpg">
</p>
<p>In today's fast-paced world, mental health issues have become very prevalent. People are constantly suffering from illnesses like depression and anxiety disorders. And because of their busy schedule or ignorance of the symptoms of such disease, people often avoid consulting a psychologist. In this project, I have built an AI system that utilizes the teachings of Gautaum Buddha and the knowledge of modern psychology to answer people's queries and provide concrete solutions for permanently fixing mental health issues. </p>
<h2>Libraries Used</h2>
<ul>
  <li>LangChain</li>
  <li>Hugging Face</li>
  <li>ChainLit</li>
</ul>
<h2>Methodology</h2>
<p>
The "Gautaum Buddha Psychologist ChatBot" functions on the retrieval augmented generation (RAG) framework at its core to provide solutions to users' queries. The heart of this system is the zephyr-7b-beta large language model. Following the RAG framework and prompt engineering techniques, a custom-made prompt is provided to the model. This prompt includes the question asked by the user and the context related to the question. 
</p>
<p>
The source of the provided context is the book named The Heart of the Buddha's Teaching. This book is in the form of embeddings generated by sentence-transformer stored as vector embeddings. Using the similarity search algorithm, the correct context is searched out and then included in the prompt, which is provided as input to the model. The chainlit framework is used here to build the chatbot UI and other chatting functionality. Using the chainlit UI, the user can easily communicate with the model and get accurate answers within a few seconds.
</p>
<h2>Zephyr 7B Beta</h2>
<p align="center">
<img src="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png">
</p>
<p>
Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO). We found that removing the in-built alignment of these datasets boosted performance on MT Bench and made the model more helpful. However, this means that model is likely to generate problematic text when prompted to do so and should only be used for educational and research purposes. 
</p>
<h2>Demo Video</h2>
https://github.com/NavinBondade/Gautama-Buddha-Psychologist-Chatbot/assets/43030152/afd9c216-3f1d-4c6b-ac1c-9386c8475288
<h2>Question Answered</h2>
<p>Below, I have asked the model how to keep mind calm? The model has successfully generated a very concise and accurate answer. The model has suggested to the user a five-step approach to calm his mind.</p>
<p align="center">
<img src="Gautama Buddha Psychologist Chatbot/result/r2.png">
</p>
<h2>Conversation With Psychologist</h2>
<p>Here, I carried out some conversations with the model as a person who has some mental issues. I asked the model a few questions and requested to elaborate on its generated answer. All the responses generated by the model were very much aligned with Gautum Buddhas's teaching. They were rich with knowledge and had practical solutions. In testing, the model had not hallucinated or tried to make up things independently.</p>
<p align="center">
<img src="Gautama Buddha Psychologist Chatbot/result/result.png">
</p>
<h2>Memory</h2>
<p>The model is also capable of remembering previous conversations and using them to understand the questions asked by the user accurately. In the below example, I asked the model "how to cure depression" and later asked "how long it will take to cure" without mentioning depression in the second question; the model correctly understood the user was talking about depression and generated a precise answer.</p>
<p align="center">
<img src="Gautama Buddha Psychologist Chatbot/result/memory.jpg">
</p>
<h2>Conclusion</h2>



