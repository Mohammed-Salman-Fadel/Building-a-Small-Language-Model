# Building a Large Language Model (_from scratch_)

This project was inspired based on the book "**Build a Large Language Model (from scratch)** by _Sebastian Raschka_".

<img src="assets/images/book_cover.jpg" width="200" />

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

## Project Layout

- `learning/` contains the chapter notebooks and learning material.
- `small_llm/` contains the actual implementation.
- `scripts/` contains the main training commands.
- `artifacts/` stores saved model checkpoints.
- `data/` stores cached datasets.

## Project Guide

To get started with the project, simply run the following to download all dependencies:

```cmd
pip install -r requirements.txt
```

The following line will immediately begin the training process on the project gutenberg dataset:

```cmd
python scripts\train_gutenberg.py
```

To finetune the model:

```cmd
python scripts\finetune_chat.py
```

And finally, you can start the UI and communicate with the result of the model training:

```cmd
chainlit run app.py
```

![image](../Images/Build-LLMS-from-scratch.png)
This is the structure by which the project follows, each divided into 3 stages, the sub-stages are seperated by folders specifying which chapters are dealing with which stage.

<strong>IMPORTANT NOTE:</strong> Some files are too large to be attached into this repository (such as the GPT-2 weights), therefore I have provided google drive links to those, if anyone is interested in directly accessing those files. It is also important to mention that if the code was ran correctly on your device, there is no need to manually grab those files, as it will be automatically downloaded from the source code itself.

<br>

### Interface for fine-tuned results

1. **Classifier:** we used the **_gradio_** library which can be found in the same file as the rest of the classifier code. Therefore simply run the notebook and an interface will appear at the end.

2. **Personal Assistant:** we use the **_chainlit_** library to visualize the chat interface with the model. This can be found in a file called 'app.py', simply run:
   <br> `pip install chainlit` then use `chainlit run app.py` to run the file.
