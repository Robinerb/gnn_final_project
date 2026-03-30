import torch
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from model import MolecularVAE
from preprocess import prepare_data, encode_strings

def get_latent_vector(model, tensor):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(tensor.unsqueeze(0))
    return mu

def decode_latent(model, z, vocab, max_len, is_selfies=False):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        z_hidden = model.latent_to_hidden(z).unsqueeze(0)
        decoder_input = torch.zeros(1, max_len, 128) 
        logits, _ = model.decoder_gru(decoder_input, z_hidden)
        logits = model.fc_out(logits)
        token_ids = torch.argmax(logits, dim=-1).squeeze(0)
        
        tokens = [inv_vocab[idx.item()] for idx in token_ids if idx.item() != 0]
        res = "".join(tokens)
        return sf.decoder(res) if is_selfies else res

def run_interpolation(n_steps=5):
    df, s_vocab, sf_vocab = prepare_data()
    max_s = max(df['smiles'].apply(len))
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))

    # load models
    sm_model = MolecularVAE(len(s_vocab), max_s)
    sm_model.load_state_dict(torch.load("checkpoints/smiles_vae_full.pth"))
    sf_model = MolecularVAE(len(sf_vocab), max_sf)
    sf_model.load_state_dict(torch.load("checkpoints/selfies_vae_full.pth"))

    # pick two molecules (start and end)
    idx1, idx2 = 0, 1
    s1 = encode_strings([df['smiles'][idx1]], s_vocab, max_s, False)
    s2 = encode_strings([df['smiles'][idx2]], s_vocab, max_s, False)
    sf1 = encode_strings([df['selfies'][idx1]], sf_vocab, max_sf, True)
    sf2 = encode_strings([df['selfies'][idx2]], sf_vocab, max_sf, True)

    z_s1, z_s2 = get_latent_vector(sm_model, s1[0]), get_latent_vector(sm_model, s2[0])
    z_sf1, z_sf2 = get_latent_vector(sf_model, sf1[0]), get_latent_vector(sf_model, sf2[0])

    print(f"Interpolating between: {df['smiles'][idx1]} and {df['smiles'][idx2]}")
    
    for i in range(n_steps):
        t = i / (n_steps - 1)
        # linear interpolation
        z_interp_s = z_s1 * (1 - t) + z_s2 * t
        z_interp_sf = z_sf1 * (1 - t) + z_sf2 * t
        
        res_s = decode_latent(sm_model, z_interp_s, s_vocab, max_s, False)
        res_sf = decode_latent(sf_model, z_interp_sf, sf_vocab, max_sf, True)
        
        print(f"Step {i} | SMILES: {res_s[:30]}... | SELFIES: {res_sf[:30]}...")

if __name__ == "__main__":
    run_interpolation()