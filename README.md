# Building a Large Language Model *from scratch*
This project is based on the book "**Build a Large Language Model (from scratch)** by *Sebastian Raschka*". 

📝 The project's objectives includes the following:
- A Working GPT-style Large Language Model
- End-to-End Pretraining Pipeline
- Fine-Tuned Model for Practical Tasks
<br>



**Components:**
1. Tokenization
2. Embeddings Model
3. Multi-Head Self- Attention Mechanism
4. Transformer Block
5. Decoding Loop
6. Fine-tuned classifier
7. Fine-tuned personal assistant



### TL;DR: The code implementation of each chapter can be found on the respective labelled notebooks, however if you want to dive directly into the final product, the final GPT model is implemented on the "GPTModel" notebook. Run the "ch06" notebook to access the fine-tuned classifier.

---
## Project Guide
![image](/images/Build-LLMS-from-scratch.png)
This is the structure by which the project follows, each divided into 3 stages, the sub-stages are seperated by folders specifying which chapters are dealing with which stage.

<strong>IMPORTANT NOTE:</strong> Some files are too large to be attached into this repository (such as the GPT-2 weights), therefore I have provided google drive links to those, if anyone is interested in directly accessing those files. It is also important to mention that if the code was ran correctly on your device, there is no need to manually grab those files, as it will be automatically downloaded from the source code itself.

<i>Google Drive Link:</i> https://drive.google.com/drive/folders/1oZ_Ih78TdHixmtUB3Hnr34c6uxGkmfzw?usp=sharing


### Interface for fine-tuned results
1. **Classifier:** we used the ***gradio*** library which can be found in the same file as the rest of the classifier code. Therefore simply run the notebook and an interface will appear at the end.

2. **Personal Assistant:** we use the ***chainlit*** library to visualize the chat interface with the model. This can be found in a file called 'app.py', simply run:
<br> `pip install chainlit` then use `chainlit run app.py` to run the file.