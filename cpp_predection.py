import os
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"
import re
import torch
import time
from sklearn.preprocessing import StandardScaler
import pickle
from torch import nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set memory optimization environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# %%
with open('./cpp_code.txt', 'r', encoding='utf-8') as f:  # Changed input file
    content = f.read()

# %%
# C++ specific preprocessing
def preprocess_cpp_code(code):
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Normalize whitespace but preserve basic structure
    code = re.sub(r'\s+', ' ', code)
    # Add space around operators and brackets for better tokenization
    code = re.sub(r'([{}();,<>=+\-*/&|!])', r' \1 ', code)
    # Normalize whitespace again
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

content = preprocess_cpp_code(content)
print(content[:100])

# %%
# Tokenize C++ code
tokens = content.split(" ")
tokens = [token for token in tokens if token.strip()]  # Remove empty tokens

# Build vocabulary
word_dict = {}
for token in tokens:
    if token in word_dict:
        word_dict[token] += 1
    else:
        word_dict[token] = 1

# %%
print(f"First 20 tokens: {tokens[:20]}")

# %%
print(f"Length of the vocabulary is: {len(word_dict)}")

# %%
sorted_items_asc = sorted(word_dict.items(), key=lambda item: item[1])
print("Least common tokens: \n")
for x in range(min(10, len(sorted_items_asc))):
    print(f"{x+1}. {sorted_items_asc[x][0]} Count: {sorted_items_asc[x][1]}")

# %%
sorted_items_dsc = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
print("Most common tokens: \n")
for x in range(min(10, len(sorted_items_dsc))):
    print(f"{x+1}. {sorted_items_dsc[x][0]} Count: {sorted_items_dsc[x][1]}")

# %%
# Build vocabulary mappings
stoi = {'_': 0}  # Padding token
itos = {0: '_'}

# Add tokens to vocabulary
for i, token in enumerate(word_dict.keys(), 1):
    stoi[token] = i
    itos[i] = token

print(f"Vocabulary size: {len(stoi)}")

# %%
# Prepare training data with C++ context - REDUCED CONTEXT SIZE
block_size = 4  # Reduced from 8 to 4 for memory
X, Y = [], []

for i in range(len(tokens) - block_size):
    context = [stoi.get(tokens[j], 0) for j in range(i, i + block_size)]
    target = stoi.get(tokens[i + block_size], 0)

    X.append(context)
    Y.append(target)

    # Print first few examples
    if i < 5:
        context_tokens = [itos.get(idx, '?') for idx in context]
        print(f"{' '.join(context_tokens)} ---> {itos.get(target, '?')}")

# %%
# Limit dataset size for training and move to GPU - REDUCED DATASET SIZE
max_samples = min(65536, len(X))  # Reduced from 262144 to 65536
X = torch.tensor(X[:max_samples], device=device)
Y = torch.tensor(Y[:max_samples], device=device)
print(f"Training data shape: {X.shape}")
print(f"Data device: {X.device}")

# %%
def plot_embeddings(selected_tokens, embeds, mode="before"):
    # Filter tokens that exist in vocabulary
    valid_tokens = [token for token in selected_tokens if token in stoi]
    if not valid_tokens:
        print("No valid tokens found for visualization")
        return

    indices = [stoi[token] for token in valid_tokens]
    indices = torch.tensor(indices, device=device)

    if mode == "before":
        selected_embeddings = embeds(indices).detach().cpu().numpy()
    else:
        selected_embeddings = embeds

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(valid_tokens)-1))
    reduced = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(10, 8))
    for i, token in enumerate(valid_tokens):
        x, y = reduced[i, 0], reduced[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.02, y + 0.02, token)

    plt.title(f"t-SNE Visualization of Learned Token Embeddings ({mode} training)")

    # Save plot as PNG instead of showing
    filename = f"plots/embeddings_{mode}_training.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved {filename}")

# %%
embeds = nn.Embedding(len(stoi), 32).to(device)
# C++ specific tokens for visualization
selected_tokens = ['int', 'void', 'for', 'while', 'if', 'else', 'return', 'cout', 'cin', 'main']
plot_embeddings(selected_tokens, embeds)

# %%
class NextToken(nn.Module):
    def __init__(self, vocab_size, context_size=4, embed_dim=32):  # REDUCED SIZES
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        # REDUCED HIDDEN LAYER SIZES
        self.hidden_layer1 = nn.Linear(input_dim, 256)  # Reduced from 512
        self.dropout1 = nn.Dropout(0.4)
        self.hidden_layer2 = nn.Linear(256, 256)  # Reduced from 512
        self.dropout2 = nn.Dropout(0.4)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h1 = F.relu(self.hidden_layer1(embeds))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.hidden_layer2(h1))
        h2 = self.dropout2(h2)
        logits = self.output(h2)
        return logits

# %%
model = NextToken(len(stoi), block_size, embed_dim=32).to(device)  # Reduced embed_dim
# REMOVED torch.compile to save memory
print(f"Model moved to: {next(model.parameters()).device}")

# %%
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print("Model Summary:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<20} {params:,} parameters")
    print(f"\nTotal trainable parameters: {count_params(model):,}\n")

# %%
def generate_code(model, itos, stoi, existing_code, block_size=4, max_len=50):  # Updated block_size
    model.eval()

    # Preprocess existing code
    existing_code = preprocess_cpp_code(existing_code)
    code_tokens = existing_code.split(" ")
    code_tokens = [token for token in code_tokens if token.strip()]

    # Build context
    if len(code_tokens) >= block_size:
        context_tokens = code_tokens[-block_size:]
    else:
        context_tokens = ['_'] * (block_size - len(code_tokens)) + code_tokens

    context = [stoi.get(token, 0) for token in context_tokens]

    generated_tokens = []
    with torch.no_grad():
        for i in range(max_len):
            x = torch.tensor(context, device=device).view(1, -1)
            y_pred = model(x)
            probs = torch.softmax(y_pred, dim=-1)
            ix = torch.argmax(probs, dim=-1).item()

            next_token = itos.get(ix, '<?>')
            generated_tokens.append(next_token)

            # Stop at reasonable points in code
            if next_token in [';', '}', '{'] and len(generated_tokens) > 10:
                break

            context = context[1:] + [ix]

    model.train()

    # Format the output
    result = ' '.join(generated_tokens)
    # Basic formatting cleanup
    result = re.sub(r'\s+([,;{}()])', r'\1', result)
    result = re.sub(r'([,;{}()])\s+', r'\1 ', result)

    return result

# %%
print_model_summary(model)

# %%
sentence = generate_code(model, itos, stoi, "int main", block_size)
print(f"Generated code before training: \n{sentence}")

# %%
# Split data and ensure it's on the correct device
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print(f"Training data on: {X_train.device}")
print(f"Validation data on: {X_val.device}")

# %%
def train_model(model, X_train, Y_train, X_val, Y_val, lr, epochs=300, batch_size=512, wd=1e-2, print_every=100):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, times, validation_losses = [], [], []

    # REDUCED BATCH SIZE for GPU
    if torch.cuda.is_available():
        batch_size = min(batch_size, 512)  # Reduced from 4096 to 512

    print(f"Training with batch size: {batch_size}")

    for e in range(epochs):
        start = time.time()
        total_loss = 0
        n_batches = 0

        # Training with gradient accumulation for memory efficiency
        model.train()
        opt.zero_grad()

        for i in range(0, X_train.shape[0], batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = Y_train[i:i+batch_size]

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch) / 2  # Scale loss for gradient accumulation

            loss.backward()

            # Gradient accumulation: update every 2 batches
            if (i // batch_size) % 2 == 1 or i + batch_size >= X_train.shape[0]:
                opt.step()
                opt.zero_grad()

                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_loss += loss.item() * 2  # Unscale for logging
            n_batches += 1

        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)

        # Validation with smaller batches to avoid OOM
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                x_val_batch = X_val[i:i+batch_size]
                y_val_batch = Y_val[i:i+batch_size]
                logits_val = model(x_val_batch)
                val_loss += loss_fn(logits_val, y_val_batch).item()
                val_batches += 1

        validation_losses.append(val_loss / val_batches)
        epoch_time = time.time() - start
        times.append(epoch_time)

        print(f"Epoch {e+1:4d} | Loss: {avg_loss:.4f} | Val Loss: {val_loss/val_batches:.4f} | Time: {epoch_time:.2f}s")

    return train_losses, validation_losses, times

# %%
# Clear GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Starting training on GPU..." if torch.cuda.is_available() else "Starting training on CPU...")

# REDUCED EPOCHS and BATCH SIZE
train_losses, validation_losses, times = train_model(
    model, X_train, Y_train, X_val, Y_val,
    lr=1e-4, epochs=26, batch_size=256, wd=0.01, print_every=50  # Reduced epochs and batch size
)

# Save model (make sure to move to CPU for saving if needed for compatibility)
model_cpu = model.to('cpu')
torch.save(model_cpu.state_dict(), "cpp_code_model_gpu.pth")
model = model.to(device)  # Move back to device

print("Training completed and model saved!")

# %%
# Test code generation
existing_code = "for (int i = 0"
generated = generate_code(model, itos, stoi, existing_code, block_size)
print(f"Existing code: {existing_code}")
print(f"Generated continuation: {generated}")
print(f"Complete: {existing_code + generated}")

# %%
# Visualize embeddings after training
embedding_matrix = model.embedding.weight.detach().cpu().numpy()
selected_tokens = ['int', 'void', 'for', 'while', 'if', 'else', 'return', 'cout', 'cin', 'main']
plot_embeddings(selected_tokens, embedding_matrix, mode="after")

# %%
# Plot training curves and save as PNG
epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='-')
plt.plot(epochs, validation_losses, label='Validation Loss', color='red', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss for C++ Code Model (GPU)")
plt.legend()
plt.grid(True)

# Save training curve as PNG
plt.savefig("plots/training_curves.png", dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("Saved plots/training_curves.png")

# %%
# Print GPU memory stats if available
if torch.cuda.is_available():
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Save model and vocabulary
with open("cpp_code_model_gpu.pkl", "wb") as model_file:
    pickle.dump(model_cpu, model_file)  # Save CPU model for compatibility

# Save vocabulary
with open('cpp_vocabulary.pkl', 'wb') as f:
    pickle.dump(stoi, f)
    pickle.dump(itos, f)

print("Model and vocabulary saved successfully!")

# Function to load model with GPU support
def load_model_gpu(model_path, vocab_size, block_size=4, embed_dim=32):  # Updated default sizes
    model = NextToken(vocab_size, block_size, embed_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Additional plot: Generate and save a comparison plot
plt.figure(figsize=(12, 4))

# Plot 1: Training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, validation_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

# Plot 2: Training time per epoch
plt.subplot(1, 2, 2)
plt.plot(epochs, times, label='Time per epoch', color='green')
plt.xlabel('Epochs')
plt.ylabel('Time (seconds)')
plt.title('Training Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/training_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/training_summary.png")

print("All plots have been saved as PNG files in the 'plots' directory!")
