import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10


class Seq2SeqModel(nn.Module):
    """
    An Seq2Seq model
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)
        decoder_outputs, _ = self.decoder(
            encoder_outputs,
            encoder_hidden,
            target,
        )
        return decoder_outputs
    
    def get_attention(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)
        _, attention = self.decoder(
            encoder_outputs,
            encoder_hidden,
            target,
        )
        return attention


class EncoderRNN(nn.Module):
    """
    An RNN encoder
    """
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    """
    An RNN decoder
    """
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None):
        """
        Here the model makes a prediction autoregressively.
        You have to handle the behaviours for when teacher forcing is enabled/disabled.
        You may assume that when target is None, teacher forcing is disabled.
        """
        batch_size = len(encoder_outputs)
        decoder_outputs = []
        decoder_hidden = encoder_hidden
        

        if target is not None:
            # Teacher forcing: Use target as the next input
            
            for t in range(batch_size):
                input_t = target[:,t]
                output, decoder_hidden = self.forward_step(input_t, decoder_hidden)
                decoder_outputs.append(output)
        else:
            # Without teacher forcing: Use its own predictions as the next input
            input_t = SOS_TOKEN * torch.ones(batch_size, 1, dtype=torch.long)
            for _ in range(batch_size):
                output, decoder_hidden = self.forward_step(input_t, decoder_hidden)
                decoder_outputs.append(output)
                input_t = output.argmax(dim=-1)  # Use the highest logit as the next input

        # Concatenate outputs along the time dimension
        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # Shape: (batch_size, seq_len, output_size)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from Vaswani et al. (2017).
    Implementation from: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    Reference: Vaswani et al. (2017) https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GPTBlock(nn.Module):
    """ A GPT-style block. """
    def __init__(self, embed_dim, num_heads, widening_factor, dropout=0.1):
        super(GPTBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.dense_1 = nn.Linear(embed_dim, embed_dim * widening_factor)
        self.dense_2 = nn.Linear(embed_dim * widening_factor, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, input, target=None):
        normed_input = self.ln_1(input)
        if target is None:
            # Use Self-attention for encoder
            att_out, attn = self.attention(normed_input, normed_input, normed_input, is_causal=False)
        else:
            # Use Cross-attention for decoder
            mask = nn.Transformer.generate_square_subsequent_mask(
                sz=input.shape[1],
                device=input.device,
            )
            att_out, attn = self.attention(target, target, normed_input, is_causal=True, attn_mask=mask)
        input = att_out + input
        normed_input = self.gelu(self.dense_1(self.ln_2(input)))
        out = input + self.dense_2(normed_input)
        return out, attn


class EncoderTransformer(nn.Module):
    """
    A transformer encoder.
    """
    def __init__(self, input_size, num_blocks, embed_dim, num_heads, widening_factor=4, dropout=0.1, max_len=10):
        super(EncoderTransformer, self).__init__()

        self.embedding = nn.Embedding(input_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout, max_len)
        gpt_blocks = []
        ln_layers = []
        for block_i in range(num_blocks):
            gpt_blocks.append(
                GPTBlock(embed_dim, num_heads, widening_factor, dropout)
            )
            ln_layers.append(
                nn.LayerNorm(embed_dim)
            )
        self.gpt_blocks = nn.ModuleList(gpt_blocks)
        self.ln_layers = nn.ModuleList(ln_layers)

    def forward(self, input):
        attns = []
        out = self.embedding(input)
        out = self.pe(out)
        for gpt_layer, ln_layer in zip(self.gpt_blocks, self.ln_layers):
            out, attn = gpt_layer(out)
            out = ln_layer(out)
            attns.append(attn.detach().cpu())
        return out, attns


class DecoderTransformer(nn.Module):
    """
    A transformer decoder.
    """
    def __init__(self, output_size, num_blocks, embed_dim, num_heads, widening_factor=4, dropout=0.1, max_len=10):
        super(DecoderTransformer, self).__init__()

        self.embedding = nn.Embedding(output_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout, max_len)
        gpt_blocks = []
        ln_layers = []
        for block_i in range(num_blocks):
            gpt_blocks.append(
                GPTBlock(embed_dim, num_heads, widening_factor, dropout)
            )
            ln_layers.append(
                nn.LayerNorm(embed_dim)
            )
        self.gpt_blocks = nn.ModuleList(gpt_blocks)
        self.ln_layers = nn.ModuleList(ln_layers)
        self.out = nn.Linear(embed_dim, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None):
        """
        Here the model makes a prediction autoregressively.
        You have to handle the behaviours for when teacher forcing is enabled/disabled.
        You may assume that when target is None, teacher forcing is disabled.

        NOTE: encoder_hidden should not be used
        """

        batch_size = len(encoder_outputs)
        decoder_outputs = []
        attns = None

        # ============================================================
        # TODO: Complete this forward loop.
        #       Take the entry with the largest logits as the prediction
        #       when teacher forcing is disabled
        if target is not None:
            # TODO: Teacher forcing: Feed the target as the next input
            pass
        else:
            # TODO: Without teacher forcing: use its own predictions as the next input
            with torch.no_grad():
                for i in range(MAX_LENGTH):
                    pass

        # ============================================================
        return decoder_outputs, attns

    def forward_step(self, input, target):
        attns = []
        out = self.embedding(input)
        out = self.pe(out)
        for gpt_layer, ln_layer in zip(self.gpt_blocks, self.ln_layers):
            out, attn = gpt_layer(out, target)
            out = ln_layer(out)
            attns.append(attn.detach().cpu())
        out = self.out(out)
        out = F.log_softmax(out, dim=-1)
        return out, attns