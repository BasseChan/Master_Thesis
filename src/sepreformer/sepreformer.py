import sys
sys.path.append('src/sepreformer/SepReformer')
import torch
from loguru import logger
logger.remove()
import yaml
from src.sepreformer.SepReformer.models.SepReformer_Base_WSJ0.model import Model
import numpy as np
import librosa


class SepReformer(object):
    def __init__(self):
        
        with open("src/sepreformer/SepReformer/models/SepReformer_Base_WSJ0/configs.yaml", 'r') as yaml_file:
            self.config = yaml.full_load(yaml_file)["config"]

        self.gpuid = tuple(map(int, self.config["engine"]["gpuid"].split(',')))
        self.device = torch.device(f'cuda:{self.gpuid[0]}')
        self.model = Model(**self.config["model"]).to(self.device)
        self.fs = self.config["dataset"]["sampling_rate"]
        self.stride = self.config["model"]["module_audio_enc"]["stride"]

        checkpoint_dict = torch.load('src/sepreformer/SepReformer/models/SepReformer_Base_WSJ0/log/scratch_weights/epoch.0180.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False) # Depend on weight file's key!!

    def __call__(self, mixture: np.ndarray | torch.Tensor | str) -> list[torch.Tensor]:
        self.model.eval()
        
        if isinstance(mixture, str):
            mixture, _ = librosa.load(mixture, sr=self.fs)
        if isinstance(mixture, np.ndarray):
            mixture = torch.tensor(mixture, dtype=torch.float32)
        mixture = mixture.unsqueeze(0)

        remains = mixture.shape[-1] % self.stride
        if remains != 0:
            padding = self.stride - remains
            mixture_padded = torch.nn.functional.pad(mixture, (0, padding), "constant", 0)
        else:
            mixture_padded = mixture

        with torch.inference_mode():
            nnet_input = mixture_padded.to(self.device)
            estim_src, _ = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
            mixture = torch.squeeze(mixture).cpu().numpy()

        return [torch.squeeze(estim_src[i][...,:mixture.shape[-1]]).cpu() for i in range(self.config['model']['num_spks'])]