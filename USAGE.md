# How To Use This Project

## Project Layout

- `learning/` contains the chapter notebooks and learning material.
- `small_llm/` contains the actual implementation.
- `scripts/` contains the main training commands.
- `artifacts/` stores saved model checkpoints.
- `data/` stores cached datasets.

## 1. Install

```powershell
python -m pip install -r requirements.txt
```

## 2. Train The Base Language Model

```powershell
python scripts\train_gutenberg.py
```

This trains on Project Gutenberg and saves checkpoints under:

- `artifacts/checkpoints/gutenberg/latest.pth`
- `artifacts/checkpoints/gutenberg/best.pth`

## 3. Fine-Tune It For Chat

```powershell
python scripts\finetune_chat.py
```

This saves the chat model to:

- `artifacts/finetuned/gutenberg-chat.pth`

## 4. Talk To The Model

```powershell
chainlit run app.py
```

If a chat-tuned checkpoint exists, the app will use it.
If not, it will fall back to the latest saved base model.

## 5. Learning Material

If you want the tutorial side of the project, open:

- `learning/main.ipynb`
- `learning/chapters/Stage1/`
- `learning/chapters/Stage2/`
- `learning/chapters/Stage3/`
