import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE_QED(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim=128, hidden_dim=256, latent_dim=64):
        super(MolecularVAE_QED, self).__init__()
        self.max_len = max_len
        
        # standard Encoder layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # standard Decoder layers
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # MLP to estimate QED (0.0 to 1.0)
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Maps input tokens to latent distribution parameters."""
        embedded = self.embedding(x)
        _, hidden = self.encoder_gru(embedded)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z):
        """Converts a latent vector back into string logits."""
        z_hidden = self.latent_to_hidden(z).unsqueeze(0)
        dummy_input = torch.zeros(z.size(0), self.max_len, 128).to(z.device)
        output, _ = self.decoder_gru(dummy_input, z_hidden)
        return self.fc_out(output)

    def reparameterize(self, mu, logvar):
        """Samples from the latent distribution for backpropagation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Joint training: reconstruction and property prediction."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # reconstruct the original string
        embedded = self.embedding(x)
        z_hidden = self.latent_to_hidden(z).unsqueeze(0)
        output, _ = self.decoder_gru(embedded, z_hidden)
        recon_logits = self.fc_out(output)
        
        # predict the QED score
        predicted_qed = self.property_predictor(z)
        
        return recon_logits, mu, logvar, predicted_qed
    
    
def joint_vae_loss(recon_x, x, mu, logvar, pred_qed, target_qed):
        # reconstruction Loss (Cross Entropy)
        recon_loss = F.cross_entropy(recon_x.transpose(1, 2), x, reduction='sum')

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Property Prediction Loss (MSE)
        property_loss = F.mse_loss(pred_qed, target_qed.view(-1, 1), reduction='sum')

        return recon_loss + kl_loss + (10.0 * property_loss)