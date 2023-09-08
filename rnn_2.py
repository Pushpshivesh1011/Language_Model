{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyOqq/jqBkxWrG0QdMhnszK5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Pushpshivesh1011/Language_Model/blob/main/rnn_2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "special_character = 'X'\n",
        "input_file_path = '/content/only_bengali.txt'\n",
        "output_file_path = 'special.txt'\n",
        "\n",
        "with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:\n",
        "    output_file.writelines([special_character + line for line in input_file])\n",
        "\n",
        "print(\"Special character added to the beginning of each sentence.\")\n"
      ],
      "metadata": {
        "id": "q7ZCPv5TofwH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install nltk\n"
      ],
      "metadata": {
        "id": "DmgsBnHqQDty"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "from nltk.tokenize import word_tokenize\n",
        "from collections import Counter\n",
        "import nltk\n",
        "nltk.download('punkt')"
      ],
      "metadata": {
        "id": "SfzZEZ1zofyQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "with open('/content/special.txt', 'r', encoding='utf-8') as file:\n",
        "    sentences = file.readlines()\n",
        "tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]"
      ],
      "metadata": {
        "id": "K9bGq4Niof2_"
      },
      "execution_count": 71,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "word_counts = Counter([word for sentence in tokenized_sentences for word in sentence])\n",
        "vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}\n",
        "vocab['<UNK>'] = len(vocab)\n",
        "vocab_size = len(vocab)\n",
        "print(f\"Vocabulary size: {vocab_size}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JUKNtKN0sxPX",
        "outputId": "8c179ba9-3883-42e3-dec8-5c767a86a049"
      },
      "execution_count": 72,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Vocabulary size: 45263\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "max_seq_length = 50\n",
        "\n",
        "padded_sequences = torch.zeros((len(tokenized_sentences), max_seq_length), dtype=torch.long)\n",
        "for i, sentence in enumerate(tokenized_sentences):\n",
        "    sequence = [vocab.get(word, vocab['<UNK>']) for word in sentence[:max_seq_length]]\n",
        "    padded_sequences[i, :len(sequence)] = torch.tensor(sequence)\n"
      ],
      "metadata": {
        "id": "iESESpx0sxST"
      },
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_ratio = 0.8\n",
        "train_size = int(train_ratio * len(sentences))\n",
        "train_data = padded_sequences[:train_size]\n",
        "test_data = padded_sequences[train_size:]\n",
        "\n",
        "total_words_train = sum(len(sentence) for sentence in train_data)\n",
        "total_words_test = sum(len(sentence) for sentence in test_data)\n",
        "\n",
        "print(f\"Total words in training dataset: {total_words_train}\")\n",
        "print(f\"Total words in testing dataset: {total_words_test}\")\n",
        "\n",
        "\n",
        "train_file_path = \"train_data.txt\"\n",
        "test_file_path = \"test_data.txt\"\n",
        "\n",
        "with open(train_file_path, \"w\") as train_file:\n",
        "    for sentence in train_data:\n",
        "        sentence_text = \" \".join(map(str, sentence))\n",
        "        train_file.write(sentence_text + \"\\n\")\n",
        "\n",
        "\n",
        "with open(test_file_path, \"w\") as test_file:\n",
        "    for sentence in test_data:\n",
        "        sentence_text = \" \".join(map(str, sentence))\n",
        "        test_file.write(sentence_text + \"\\n\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xu64daZEtEZ8",
        "outputId": "baa4aef2-9856-486e-f747-bb81cef47114"
      },
      "execution_count": 74,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total words in training dataset: 842200\n",
            "Total words in testing dataset: 210550\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 64\n",
        "input_dim = 128\n",
        "hidden_dim = 256\n",
        "num_layers = 2\n",
        "num_epochs = 10\n",
        "learning_rate = 0.001\n"
      ],
      "metadata": {
        "id": "YCJKyhS0tEcy"
      },
      "execution_count": 75,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(f\"Input data shape: ({max_seq_length}, {batch_size}, {input_dim})\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "16sgGA0DtQtB",
        "outputId": "8441677d-d34b-47f5-e247-4216c3396982"
      },
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Input data shape: (50, 64, 128)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)\n"
      ],
      "metadata": {
        "id": "WPz-DxjwtQv0"
      },
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class RNNModel(nn.Module):\n",
        "    def __init__(self, vocab_size, input_dim, hidden_dim, num_layers):\n",
        "        super(RNNModel, self).__init__()\n",
        "        self.embedding = nn.Embedding(vocab_size, input_dim)\n",
        "        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)\n",
        "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
        "\n",
        "    def forward(self, x):\n",
        "        embedded = self.embedding(x)\n",
        "        output, _ = self.rnn(embedded)\n",
        "        output = self.fc(output)\n",
        "        return output\n",
        "\n",
        "model = RNNModel(vocab_size, input_dim, hidden_dim, num_layers)\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n"
      ],
      "metadata": {
        "id": "B9gKjLjdtQze"
      },
      "execution_count": 78,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "train_data = train_data.to(device)\n",
        "test_data = test_data.to(device)\n"
      ],
      "metadata": {
        "id": "0EFzJahNwH3A"
      },
      "execution_count": 79,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    for batch in train_loader:\n",
        "        optimizer.zero_grad()\n",
        "        inputs = batch[0].to(device)\n",
        "        targets = inputs[:, 1:]\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "\n",
        "\n",
        "    average_loss = total_loss / len(train_loader)\n",
        "    print(f\"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {average_loss:.4f}\")\n",
        "\n",
        "model.eval()\n",
        "\n",
        "print(\"Training completed. You can print the evaluation results here.\")\n",
        "\n",
        "model_save_path = 'rnn_model.pth'\n",
        "torch.save(model.state_dict(), model_save_path)\n",
        "print(f\"Word_level_Model saved to {model_save_path}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pGYdEW9GwH5x",
        "outputId": "559dc4dc-3d01-4b64-f090-0bf72ce39666"
      },
      "execution_count": 80,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch [1/10] - Average Loss: 2.9976\n",
            "Epoch [2/10] - Average Loss: 2.5125\n",
            "Epoch [3/10] - Average Loss: 2.4331\n",
            "Epoch [4/10] - Average Loss: 2.3484\n",
            "Epoch [5/10] - Average Loss: 2.2639\n",
            "Epoch [6/10] - Average Loss: 2.1692\n",
            "Epoch [7/10] - Average Loss: 2.0684\n",
            "Epoch [8/10] - Average Loss: 1.9594\n",
            "Epoch [9/10] - Average Loss: 1.8455\n",
            "Epoch [10/10] - Average Loss: 1.7269\n",
            "Training completed. You can print the evaluation results here.\n",
            "Word_level_Model saved to rnn_model.pth\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "\n",
        "num_epochs = 10\n",
        "\n",
        "total_loss_10th_epoch = 0.0\n",
        "total_tokens_10th_epoch = 0\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "\n",
        "    for batch in test_loader:\n",
        "        optimizer.zero_grad()\n",
        "        inputs = batch[0].to(device)\n",
        "        targets = inputs[:, 1:]\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "        total_tokens = targets.numel()\n",
        "        print(len(total_tokens))\n",
        "        total_tokens_10th_epoch += total_tokens\n",
        "\n",
        "    if epoch == 9:\n",
        "        total_loss_10th_epoch = total_loss\n",
        "\n",
        "average_loss_10th_epoch = total_loss_10th_epoch / len(train_loader)\n",
        "print(f\"Cross-Entropy Loss in the 10th epoch: {average_loss_10th_epoch:.4f}\")\n",
        "\n",
        "normalized_loss = total_loss_10th_epoch / total_tokens_10th_epoch\n",
        "perplexity_10th_epoch = torch.exp(torch.tensor(normalized_loss))\n",
        "print(f\"Perplexity on the 10th epoch: {perplexity_10th_epoch:.4f}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4nbPMVzv9Wiv",
        "outputId": "95002bce-ae31-4e9d-8536-2f05a899743d"
      },
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cross-Entropy Loss in the 10th epoch: 0.2951\n",
            "Perplexity on the 10th epoch: 1.0000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def remove_padding(line):\n",
        "\n",
        "    parts = line.split('\\t')\n",
        "    trimmed_parts = [part.strip() for part in parts]\n",
        "\n",
        "    cleaned_line = '\\t'.join(trimmed_parts)\n",
        "\n",
        "    return cleaned_line\n",
        "\n",
        "input_file = '/content/test_data.txt'\n",
        "output_file = 'paded_removed_word_test_data.txt'\n",
        "\n",
        "with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:\n",
        "    for line in infile:\n",
        "        cleaned_line = remove_padding(line)\n",
        "\n",
        "        outfile.write(cleaned_line + '\\n')\n",
        "\n",
        "print(\"Padding removed and saved to\", output_file)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AZfjMN8DUJW8",
        "outputId": "c6836b8f-7936-488e-8d47-feaa64ec4cd4"
      },
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Padding removed and saved to paded_removed_word_test_data.txt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def calculate_perplexity(model, dataset, vocab_size, device):\n",
        "\n",
        "    with open(dataset, 'r', encoding='utf-8') as file:\n",
        "        sentences = file.readlines()\n",
        "\n",
        "    normalized_loss = total_loss / total_tokens\n",
        "    perplexity = torch.exp(torch.tensor(normalized_loss))\n",
        "\n",
        "    return perplexity\n",
        "\n",
        "model_path = '/content/rnn_model.pth'\n",
        "dataset_path = input(\"Enter the path to the dataset file: \")\n",
        "\n",
        "perplexity = calculate_perplexity(model, dataset_path, vocab_size, device)\n",
        "\n",
        "print(f\"Perplexity on the provided dataset: {perplexity:.4f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ktXQ9lEKVnSN",
        "outputId": "7b679389-19c2-4384-be39-e82ff95f072b"
      },
      "execution_count": 83,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter the path to the dataset file: /content/paded_removed_word_test_data.txt\n",
            "Perplexity on the provided dataset: 1.0317\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "fKN93v6hYlRw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "55ocIdTlUKEF"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "1q2lhMUIUKHD"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "HBBx0UyPUKK2"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "1b3QeEyDUKPY"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "yXxu-W_6UKS1"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "kEvErlv8UKXi"
      },
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "from collections import Counter\n",
        "import string\n",
        "\n",
        "with open('/content/special.txt', 'r', encoding='utf-8') as file:\n",
        "    text = file.read()\n",
        "\n",
        "tokenized_text = list(text)\n",
        "\n",
        "vocab = {char: idx for idx, char in enumerate(set(tokenized_text))}\n",
        "vocab['<UNK>'] = len(vocab)\n",
        "\n",
        "vocab_size = len(vocab)\n",
        "print(f\"Vocabulary size: {vocab_size}\")\n",
        "\n",
        "max_seq_length = 50\n",
        "\n",
        "indexed_text = [vocab.get(char, vocab['<UNK>']) for char in tokenized_text]\n",
        "\n",
        "num_batches = len(indexed_text) // max_seq_length\n",
        "padded_sequences = torch.zeros((num_batches, max_seq_length), dtype=torch.long)\n",
        "for i in range(num_batches):\n",
        "    start_idx = i * max_seq_length\n",
        "    end_idx = start_idx + max_seq_length\n",
        "    sequence = indexed_text[start_idx:end_idx]\n",
        "    padded_sequences[i, :len(sequence)] = torch.tensor(sequence)\n",
        "\n",
        "data = padded_sequences\n",
        "\n",
        "train_ratio = 0.8\n",
        "train_size = int(train_ratio * len(data))\n",
        "\n",
        "train_data = data[:train_size]\n",
        "test_data = data[train_size:]\n",
        "\n",
        "total_chars_train = len(train_data.view(-1))\n",
        "total_chars_test = len(test_data.view(-1))\n",
        "\n",
        "print(f\"Total characters in training dataset: {total_chars_train}\")\n",
        "print(f\"Total characters in testing dataset: {total_chars_test}\")\n",
        "\n",
        "train_file_path = \"train_char_data.txt\"\n",
        "test_file_path = \"test_char_data.txt\"\n",
        "\n",
        "with open(train_file_path, \"w\") as train_file:\n",
        "    for sentence in train_data:\n",
        "        sentence_text = \" \".join(map(str, sentence))\n",
        "        train_file.write(sentence_text + \"\\n\")\n",
        "\n",
        "\n",
        "with open(test_file_path, \"w\") as test_file:\n",
        "    for sentence in test_data:\n",
        "        sentence_text = \" \".join(map(str, sentence))\n",
        "        test_file.write(sentence_text + \"\\n\")\n",
        "\n",
        "batch_size = 64\n",
        "input_dim = 128\n",
        "hidden_dim = 256\n",
        "num_layers = 2\n",
        "num_epochs = 10\n",
        "learning_rate = 0.001\n",
        "print(f\"Input data shape: ({batch_size}, {max_seq_length})\")\n",
        "\n",
        "train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)\n",
        "\n",
        "class CharRNNModel(nn.Module):\n",
        "    def __init__(self, vocab_size, input_dim, hidden_dim, num_layers):\n",
        "        super(CharRNNModel, self).__init__()\n",
        "        self.embedding = nn.Embedding(vocab_size, input_dim)\n",
        "        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)\n",
        "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
        "\n",
        "    def forward(self, x):\n",
        "        embedded = self.embedding(x)\n",
        "        output, _ = self.rnn(embedded)\n",
        "        output = self.fc(output)\n",
        "        return output\n",
        "\n",
        "model = CharRNNModel(vocab_size, input_dim, hidden_dim, num_layers)\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "train_data = train_data.to(device)\n",
        "test_data = test_data.to(device)\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    for batch in train_loader:\n",
        "        optimizer.zero_grad()\n",
        "        inputs = batch[0].to(device)\n",
        "        targets = inputs[:, 1:]\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "\n",
        "    average_loss = total_loss / len(train_loader)\n",
        "    print(f\"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {average_loss:.4f}\")\n",
        "\n",
        "model.eval()\n",
        "\n",
        "print(\"Training completed..\")\n",
        "\n",
        "model_save_path = 'char_rnn_model.pth'\n",
        "torch.save(model.state_dict(), model_save_path)\n",
        "print(f\"Char_Model saved to {model_save_path}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fVSblN4u7NBh",
        "outputId": "4b7acaac-2b2a-4136-8db9-88a806de0188"
      },
      "execution_count": 85,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Vocabulary size: 317\n",
            "Total characters in training dataset: 1662350\n",
            "Total characters in testing dataset: 415600\n",
            "Input data shape: (64, 50)\n",
            "Epoch [1/10] - Average Loss: 2.8445\n",
            "Epoch [2/10] - Average Loss: 2.2110\n",
            "Epoch [3/10] - Average Loss: 1.9814\n",
            "Epoch [4/10] - Average Loss: 1.8502\n",
            "Epoch [5/10] - Average Loss: 1.7670\n",
            "Epoch [6/10] - Average Loss: 1.7096\n",
            "Epoch [7/10] - Average Loss: 1.6667\n",
            "Epoch [8/10] - Average Loss: 1.6327\n",
            "Epoch [9/10] - Average Loss: 1.6047\n",
            "Epoch [10/10] - Average Loss: 1.5806\n",
            "Training completed..\n",
            "Char_Model saved to char_rnn_model.pth\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "\n",
        "num_epochs = 10\n",
        "\n",
        "total_loss_10th_epoch = 0.0\n",
        "total_tokens_10th_epoch = 0\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "\n",
        "    for batch in test_loader:\n",
        "        optimizer.zero_grad()\n",
        "        inputs = batch[0].to(device)\n",
        "        targets = inputs[:, 1:]\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "        total_tokens = targets.numel()\n",
        "        total_tokens_10th_epoch += total_tokens\n",
        "\n",
        "    if epoch == 9:\n",
        "        total_loss_10th_epoch = total_loss\n",
        "\n",
        "average_loss_10th_epoch = total_loss_10th_epoch / len(train_loader)\n",
        "print(f\"Cross-Entropy Loss in the 10th epoch: {average_loss_10th_epoch:.4f}\")\n",
        "\n",
        "normalized_loss = total_loss_10th_epoch / total_tokens_10th_epoch\n",
        "perplexity_10th_epoch = torch.exp(torch.tensor(normalized_loss))\n",
        "print(f\"Perplexity on the 10th epoch: {perplexity_10th_epoch:.4f}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EqfE2Wcv7NJ2",
        "outputId": "bf968cd2-2b3d-42f0-bfea-2d166703818a"
      },
      "execution_count": 86,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cross-Entropy Loss in the 10th epoch: 0.3789\n",
            "Perplexity on the 10th epoch: 1.0000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def remove_padding(line):\n",
        "\n",
        "    parts = line.split('\\t')\n",
        "    trimmed_parts = [part.strip() for part in parts]\n",
        "\n",
        "    cleaned_line = '\\t'.join(trimmed_parts)\n",
        "\n",
        "    return cleaned_line\n",
        "\n",
        "input_file = '/content/test_char_data.txt'\n",
        "output_file = 'paded_removed_char_test_data.txt'\n",
        "\n",
        "with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:\n",
        "    for line in infile:\n",
        "        cleaned_line = remove_padding(line)\n",
        "\n",
        "        outfile.write(cleaned_line + '\\n')\n",
        "\n",
        "print(\"Padding removed and saved to\", output_file)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x9i--Ac_YdZy",
        "outputId": "9eff91e5-7caf-4b41-9747-58f6a838cdea"
      },
      "execution_count": 87,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Padding removed and saved to paded_removed_char_test_data.txt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def calculate_perplexity(model, dataset, vocab_size, device):\n",
        "\n",
        "    with open(dataset, 'r', encoding='utf-8') as file:\n",
        "        sentences = file.readlines()\n",
        "\n",
        "    normalized_loss = total_loss / total_tokens\n",
        "    perplexity = torch.exp(torch.tensor(normalized_loss))\n",
        "\n",
        "    return perplexity\n",
        "\n",
        "model_path = '/content/char_rnn_model.pth'\n",
        "dataset_path = input(\"Enter the path to the dataset file: \")\n",
        "\n",
        "perplexity = calculate_perplexity(model, dataset_path, vocab_size, device)\n",
        "\n",
        "print(f\"Perplexity on the provided dataset: {perplexity:.4f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OtrQlcc7Ymyc",
        "outputId": "6b204cee-7277-4942-b9af-a4d644745ca3"
      },
      "execution_count": 88,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter the path to the dataset file: /content/paded_removed_char_test_data.txt\n",
            "Perplexity on the provided dataset: 1.0744\n"
          ]
        }
      ]
    }
  ]
}