GPT Language Model on "The Wizard of Oz" Text
=============================================

Introduction
------------

This project implements a GPT (Generative Pretrained Transformer) language model using PyTorch, trained on the text from *The Wizard of Oz*. The model is designed to generate coherent sequences of text by predicting the next character in a sequence based on previous characters. This is a simplified character-level GPT model, where each character in the text is treated as a token.

Requirements
------------

To run this project, you'll need the following dependencies:

-   Python 3.x
-   PyTorch
-   CUDA-enabled GPU (optional but recommended for faster training)
-   Other libraries: `pickle`, `argparse`, `mmap`, and `random`.

Model Overview
--------------

The architecture of the model is based on the Transformer framework, with self-attention layers, feedforward layers, and layer normalization. The model is capable of:

-   Learning from text by analyzing character sequences.
-   Predicting the next character in a sequence.
-   Generating text based on a given prompt.

Key model features:

-   **Multi-Head Attention**: This allows the model to focus on different parts of the input sequence simultaneously.
-   **Feedforward Neural Networks**: These help the model process and transform the data after the attention mechanism.
-   **Layer Normalization**: Used to stabilize training by normalizing the input to each layer.
-   **Dropout**: Helps prevent overfitting during training.

Data Preparation
----------------

The model uses *The Wizard of Oz* text for training. It treats every character in the text as a token, mapping each character to a unique integer. The text is split into:

-   **Training Set**: 80% of the text.
-   **Validation Set**: 20% of the text.

### Data Encoding and Decoding

-   **Encoding**: Characters in the text are converted to their corresponding integer values using a dictionary (`string_to_int`).
-   **Decoding**: The model's predictions (which are integers) are converted back to characters using another dictionary (`int_to_string`).

Model Training
--------------

### Hyperparameters:

-   `batch_size = 32`
-   `block_size = 128` (The number of characters in each input sequence)
-   `n_embd = 384`, `n_head = 4`, `n_layer = 4` (Embedding size, number of attention heads, and number of layers)
-   `learning_rate = 3e-4`
-   `dropout = 0.2`

### Training Process:

1.  The training loop runs for `max_iters = 3000` iterations.
2.  For every batch, the model calculates loss using cross-entropy.
3.  An AdamW optimizer is used to minimize the loss.
4.  The model evaluates training and validation losses periodically to ensure it's learning properly.

### Loss Estimation

The model calculates loss on both training and validation sets every `eval_iters = 50` iterations, which helps track overfitting or underfitting.

Text Generation
---------------

Once the model is trained, you can use it to generate text. To do so:

1.  Provide a prompt (e.g., "Hello! Can you see me?").
2.  The model predicts the next character one by one based on the prompt and continues until it reaches the specified maximum number of tokens (characters).

Running the Code
----------------

1.  Clone the repository or download the `wizard_of_oz.txt` file from the provided link.
2.  Run the script in a PyTorch-compatible environment:


```
    # bash
    python your_script_name.py
```

3.  Monitor the training process by tracking the losses and output.
4.  Use the `generate` function to generate text after training:

```
    # python

    prompt = 'Hello!'
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
    print(generated_chars)
```

Example Output
--------------

```
# text

Hello! Can you see me?
""
"So he hasken the mountain top of the Mangaboos."
"Wolld you sup sin what yo...
```

Notes
-----

-   The generated text might be nonsensical, especially in the early stages of training. With more training and parameter tuning, the quality of the generated text improves.
-   Make sure to have a GPU enabled for faster training, as transformer models can be computationally intensive.
