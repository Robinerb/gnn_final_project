import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim=128, hidden_dim=256, latent_dim=64):
        super(MolecularVAE, self).__init__()
        self.max_len = max_len
        
        # encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """Converts input sequence to latent distribution parameters."""
        embedded = self.embedding(x)
        _, hidden = self.encoder_gru(embedded)
        hidden = hidden.squeeze(0)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """VAE sampling trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_hidden = self.latent_to_hidden(z).unsqueeze(0) 
        dummy_input = torch.zeros(z.size(0), self.max_len, 128).to(z.device)
        
        output, _ = self.decoder_gru(dummy_input, z_hidden)
        logits = self.fc_out(output)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        embedded = self.embedding(x)
        z_hidden = self.latent_to_hidden(z).unsqueeze(0)
        
        output, _ = self.decoder_gru(embedded, z_hidden)
        logits = self.fc_out(output)
        
        return logits, mu, logvar