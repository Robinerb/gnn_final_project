import torch
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from model import MolecularVAE
from preprocess import prepare_data, encode_strings

def get_latent_vector(model, tensor):
    """Encodes a single molecule tensor into its latent mu vector."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(tensor.unsqueeze(0))
    return mu

def decode_to_smiles(model, z, vocab, max_len, is_selfies):
    """Decodes a latent vector back into a SMILES string."""
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        logits = model.decode(z.view(1, -1))
        token_ids = torch.argmax(logits, dim=-1).squeeze(0)
        tokens = [inv_vocab[idx.item()] for idx in token_ids if idx.item() != 0]
        gen_str = "".join(tokens)
        return sf.decoder(gen_str) if is_selfies else gen_str

def run_evaluation():
    # load Data and Models
    df, s_vocab, sf_vocab = prepare_data()
    max_s = max(df['smiles'].apply(len))
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))

    sm_model = MolecularVAE(len(s_vocab), max_s)
    sm_model.load_state_dict(torch.load("checkpoints/smiles_vae_full.pth"))
    sf_model = MolecularVAE(len(sf_vocab), max_sf)
    sf_model.load_state_dict(torch.load("checkpoints/selfies_vae_full.pth"))

    # Visual Interpolation 
    # pick two distinct molecules from subset
    idx1, idx2 = 10, 50 
    print(f"Walking between: {df['smiles'][idx1]} and {df['smiles'][idx2]}")
    
    # get vectors
    s1_t = encode_strings([df['smiles'][idx1]], s_vocab, max_s, False)
    s2_t = encode_strings([df['smiles'][idx2]], s_vocab, max_s, False)
    z_s1, z_s2 = get_latent_vector(sm_model, s1_t[0]), get_latent_vector(sm_model, s2_t[0])
    
    sf1_t = encode_strings([df['selfies'][idx1]], sf_vocab, max_sf, True)
    sf2_t = encode_strings([df['selfies'][idx2]], sf_vocab, max_sf, True)
    z_sf1, z_sf2 = get_latent_vector(sf_model, sf1_t[0]), get_latent_vector(sf_model, sf2_t[0])

    # generate the walk steps
    steps = 5
    smiles_walk = []
    selfies_walk = []
    
    for i in range(steps):
        t = i / (steps - 1)
        # SMILES path
        z_s = z_s1 * (1 - t) + z_s2 * t
        smiles_walk.append(decode_to_smiles(sm_model, z_s, s_vocab, max_s, False))
        # SELFIES path
        z_sf = z_sf1 * (1 - t) + z_sf2 * t
        selfies_walk.append(decode_to_smiles(sf_model, z_sf, sf_vocab, max_sf, True))

    # save images
    save_mol_grid(smiles_walk, "figures/miles_interpolation.png")
    save_mol_grid(selfies_walk, "figures/selfies_interpolation.png")

    # Quantitative Metrics
    print("\n--- Calculating Metrics (500 samples) ---")
    training_set = set(df['smiles'])
    sm_metrics = calculate_metrics(sm_model, s_vocab, max_s, training_set, False)
    sf_metrics = calculate_metrics(sf_model, sf_vocab, max_sf, training_set, True)
    
    print(f"SMILES Results: {sm_metrics}")
    print(f"SELFIES Results: {sf_metrics}")

def save_mol_grid(smiles_list, filename):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [m for m in mols if m is not None]
    if valid_mols:
        img = Draw.MolsToGridImage(valid_mols, molsPerRow=len(valid_mols), subImgSize=(300, 300))
        img.save(filename)
        print(f"Saved visualization to {filename}")

def calculate_metrics(model, vocab, max_len, train_set, is_selfies):
    model.eval()
    n_samples = 500
    gen_smiles = []
    inv_vocab = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        z = torch.randn(n_samples, 64)
        logits = model.decode(z)
        token_ids = torch.argmax(logits, dim=-1)
        for i in range(n_samples):
            tokens = [inv_vocab[idx.item()] for idx in token_ids[i] if idx.item() != 0]
            s = "".join(tokens)
            gen_smiles.append(sf.decoder(s) if is_selfies else s)

    valid = [s for s in gen_smiles if Chem.MolFromSmiles(s) is not None]
    unique = set(valid)
    novel = [s for s in unique if s not in train_set]
    
    return {
        "Validity": len(valid) / n_samples,
        "Uniqueness": len(unique) / len(valid) if valid else 0,
        "Novelty": len(novel) / len(unique) if unique else 0
    }

if __name__ == "__main__":
    run_evaluation()