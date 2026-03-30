import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import selfies as sf

from model_qed import MolecularVAE_QED, joint_vae_loss
from preprocess import prepare_data, encode_strings

class MolPropertyDataset(Dataset):
    def __init__(self, tensors, properties):
        self.tensors = tensors
        self.properties = torch.tensor(properties).float()
        
    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        return self.tensors[idx], self.properties[idx]

def train_qed_model(tensors, qed_scores, vocab_size, max_len, epochs=30, batch_size=64):
    model = MolecularVAE_QED(vocab_size, max_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = MolPropertyDataset(tensors, qed_scores)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = []
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_tensors, batch_qed in loader:
            optimizer.zero_grad()
            
            # forward pass
            recon_logits, mu, logvar, pred_qed = model(batch_tensors)
            
            # joint loss
            loss = joint_vae_loss(recon_logits, batch_tensors, mu, logvar, pred_qed, batch_qed)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Joint Loss: {avg_loss:.4f}")
            
    # save model and history
    torch.save(model.state_dict(), "checkpoints/selfies_vae_qed.pth")
    pd.DataFrame(history).to_csv("results/qed_training_history.csv", index=False)
    return model

if __name__ == "__main__":
    # load data
    df, s_vocab, sf_vocab = prepare_data()
    
    # get QED scores from your subset
    qed_scores = df['qed'].values
    
    # process SELFIES
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))
    sf_tensors = encode_strings(df['selfies'], sf_vocab, max_sf, is_selfies=True)
    
    # train
    model = train_qed_model(sf_tensors, qed_scores, len(sf_vocab), max_sf)