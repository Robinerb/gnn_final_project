import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import MolecularVAE, vae_loss_function
from preprocess import prepare_data, encode_strings
import selfies as sf
import pandas as pd

# dataset wrapper
class MolDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return self.tensors[idx]

def train_with_loader(tensors, vocab_size, max_len, model_name, epochs=30, batch_size=64):
    model = MolecularVAE(vocab_size, max_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(MolDataset(tensors), batch_size=batch_size, shuffle=True)
    
    # history storage
    history = []
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            loss = vae_loss_function(logits, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            
    #save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"results/{model_name.lower()}_history.csv", index=False)
    
    return model

if __name__ == "__main__":
    # load and process samples
    df, s_vocab, sf_vocab = prepare_data()
    
    # calculate dimensions based on the larger dataset
    max_s = max(df['smiles'].apply(len))
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))
    
    # encode strings to tensors
    s_tensors = encode_strings(df['smiles'], s_vocab, max_s, is_selfies=False)
    sf_tensors = encode_strings(df['selfies'], sf_vocab, max_sf, is_selfies=True)

    # train SMILES VAE
    sm_model = train_with_loader(s_tensors, len(s_vocab), max_s, "SMILES")
    torch.save(sm_model.state_dict(), "checkpoints/smiles_vae_full.pth")

    # train SELFIES VAE
    sf_model = train_with_loader(sf_tensors, len(sf_vocab), max_sf, "SELFIES")
    torch.save(sf_model.state_dict(), "checkpoints/selfies_vae_full.pth")
    