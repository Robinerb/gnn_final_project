import torch
import torch.optim as optim
import selfies as sf
from model_qed import MolecularVAE_QED
from preprocess import prepare_data, encode_strings

def optimize_molecule(idx=0, steps=50, lr=0.1, return_score=False):
    # setup
    df, s_vocab, sf_vocab = prepare_data()
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))
    
    model = MolecularVAE_QED(len(sf_vocab), max_sf)
    model.load_state_dict(torch.load("checkpoints/selfies_vae_qed.pth"))
    model.eval()

    # get initial latent vector (z)
    initial_sf = df['selfies'][idx]
    initial_qed = df['qed'][idx]
    tensor = encode_strings([initial_sf], sf_vocab, max_sf, is_selfies=True)
    
    with torch.no_grad():
        mu, _ = model.encode(tensor)
        z = mu.clone().detach().requires_grad_(True)

    print(f"Starting Molecule: {df['smiles'][idx]}")
    print(f"Initial QED: {initial_qed:.4f}")

    # gradient Ascent
    # maximise predicted_qed
    optimizer = optim.Adam([z], lr=lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        pred_qed = model.property_predictor(z)
        loss = -pred_qed
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"Step {i+1} | Predicted QED: {pred_qed.item():.4f}")

    # decode the optimized result
    inv_vocab = {v: k for k, v in sf_vocab.items()}
    with torch.no_grad():
        
        # get the final predicted score
        final_pred_qed = model.property_predictor(z).item()
        
        logits = model.decode(z)
        token_ids = torch.argmax(logits, dim=-1).squeeze(0)
        tokens = [inv_vocab[idx.item()] for idx in token_ids if idx.item() != 0]
        optimized_sf = "".join(tokens)
        optimized_smiles = sf.decoder(optimized_sf)
        
    print(f"\nOptimized SMILES: {optimized_smiles}")
    
    if return_score:
        return optimized_smiles, final_pred_qed
    return optimized_smiles

if __name__ == "__main__":
    optimize_molecule(idx=15)