import sys
sys.path.append('src/fastgeco/Fast-GeCo')

import torch
from fastgeco.model import ScoreModel
from geco.util.other import pad_spec
import os
from speechbrain.lobes.models.dual_path import Encoder, SBTransformerBlock, SBTransformerBlock, Dual_Path_Model, Decoder


class GeCo:
    def __init__(self, ckpt_path, num_spks=2, num_reversed_steps=1, reverse_starting_point=0.5, sr=8000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder, self.masknet, self.decoder = self.load_sepformer(ckpt_path, num_spks)
        self.fastgeco = self.load_fastgeco(ckpt_path)
        self.num_spks = num_spks
        self.num_reversed_steps = num_reversed_steps
        self.reverse_starting_point = reverse_starting_point
        self.sr = sr

    def load_sepformer(self, ckpt_path, num_spks=2):
        encoder = Encoder(
            kernel_size=160, 
            out_channels=256, 
            in_channels=1
        ).to(self.device)
        SBtfintra = SBTransformerBlock(
            num_layers=8,
            d_model=256,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )
        SBtfinter = SBTransformerBlock(
            num_layers=8,
            d_model=256,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )
        masknet = Dual_Path_Model(
            num_spks=num_spks,
            in_channels=256,
            out_channels=256,
            num_layers=2,
            K=250,
            intra_model=SBtfintra,
            inter_model=SBtfinter,
            norm='ln',
            linear_layer_after_inter_intra=False,
            skip_around_intra=True,
        ).to(self.device)
        decoder = Decoder(
            in_channels=256,
            out_channels=1,
            kernel_size=160,
            stride=80,
            bias=False,
        ).to(self.device)

        encoder_weights = torch.load(os.path.join(ckpt_path, 'encoder.ckpt'))
        encoder.load_state_dict(encoder_weights)
        masknet_weights = torch.load(os.path.join(ckpt_path, 'masknet.ckpt'))
        masknet.load_state_dict(masknet_weights)
        decoder_weights = torch.load(os.path.join(ckpt_path, 'decoder.ckpt'))
        decoder.load_state_dict(decoder_weights)

        encoder.eval()
        masknet.eval()
        decoder.eval()
        return encoder, masknet, decoder

    def load_fastgeco(self, ckpt_path):
        checkpoint_file = os.path.join(ckpt_path, 'fastgeco.ckpt')
        model = ScoreModel.load_from_checkpoint(
            checkpoint_file,
            batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        model.eval(no_ema=False)
        model.cuda()
        return model
    
    @torch.no_grad()
    def separate(self, mix):
        mix = mix.to(self.device)
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        est_sources = torch.cat([self.decoder(sep_h[i]).unsqueeze(-1) for i in range(self.num_spks)], dim=-1)
        est_sources = (est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]).squeeze()

        return est_sources
    
    @torch.no_grad()
    def correct(self, mix, est_sources):
        output = []
        for idx in range(self.num_spks):
            y = est_sources[:, idx].unsqueeze(0) # noisy
            m = mix.to(self.device)
            min_leng = min(y.shape[-1],m.shape[-1])
            y = y[...,:min_leng]
            m = m[...,:min_leng]
            T_orig = y.size(1)   

            norm_factor = y.abs().max()
            y = y / norm_factor
            m = m / norm_factor 
            Y = torch.unsqueeze(self.fastgeco._forward_transform(self.fastgeco._stft(y.to(self.device))), 0)
            Y = pad_spec(Y)
            M = torch.unsqueeze(self.fastgeco._forward_transform(self.fastgeco._stft(m.to(self.device))), 0)
            M = pad_spec(M)

            timesteps = torch.linspace(self.reverse_starting_point, 0.03, self.num_reversed_steps, device=Y.device)
            std = self.fastgeco.sde._std(self.reverse_starting_point*torch.ones((Y.shape[0],), device=Y.device))
            z = torch.randn_like(Y)
            X_t = Y + z * std[:, None, None, None]
            
            t = timesteps[0]
            dt = timesteps[-1]
            f, g = self.fastgeco.sde.sde(X_t, t, Y)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
            mean_x_tm1 = X_t - (f - g**2*self.fastgeco.forward(X_t, vec_t, Y, M, vec_t[:,None,None,None]))*dt #mean of x t minus 1 = mu(x_{t-1})
            sample = mean_x_tm1 
            sample = sample.squeeze()
            x_hat = self.fastgeco.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor
            new_norm_factor = x_hat.abs().max()
            x_hat = x_hat / new_norm_factor
            output.append(x_hat)
        return output
            

    def __call__(self, mix):
        separated = self.separate(mix)
        corrected = self.correct(mix, separated)
        return corrected