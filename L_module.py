import torch
import numpy as np
import torch.nn as nn
import lightning as L  
from typing import List, Tuple, Any, Dict, Optional, Mapping
import torch.nn.functional as F
from utils_tfasr import stich_by_clip



class TfarSR(L.LightningModule):
    def __init__(self, generator:nn.Module, hr_norm:nn.Module,
                 lr_norm:nn.Module, learning_rate:float, slope_fn:nn.Module,
                 river_fn:nn.Module, river_unet:nn.Module,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.normalize_hr = True
        self.input_normalizer = lr_norm
        self.ram_normalizer = hr_norm
        self.learning_rate = learning_rate
        self.slope_fn = slope_fn
        self.river_fn = river_fn
        self.river_unet = river_unet
        self.mse = nn.MSELoss()
        self.register_buffer("k_river", torch.tensor(1e-3), persistent=False)

        
        
    def forward(self, lr:torch.Tensor) -> torch.Tensor:
        bic = F.interpolate(lr, scale_factor=4, mode='bicubic' )
        sr = self.generator(bic)
        return sr

    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        return opt
    
    def training_step(self, batch:Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        lr, hr = self._prep_batch(batch= batch)
        sr = self.forward(lr)
        
        dict_loss = self._calc_loss(lr, hr, sr)
        elev_loss = dict_loss['elev_loss']
        slope_loss = dict_loss['slope_loss']
        river_loss = dict_loss['river_loss']
        t_loss = elev_loss + slope_loss + self.k_river*river_loss
        dict_loss['t_loss'] = t_loss
        self.log_dict(dict_loss, prog_bar=True, on_step=True)
        return t_loss
    
    def validation_step(self, batch:Any, *args: Any, 
        **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        lr, hr = self._prep_batch(batch= batch)
        sr = self.forward(lr)
        
        dict_loss = self._calc_loss(lr, hr, sr)
        elev_loss = dict_loss['elev_loss']
        slope_loss = dict_loss['slope_loss']
        river_loss = dict_loss['river_loss']
        v_loss = elev_loss + slope_loss + self.k_river*river_loss
        dict_loss['loss'] = v_loss
        self.log_dict({f"val_{k}": v for k, v in dict_loss.items()}, 
                      prog_bar=True, on_step=False)
        return {
            "lr": lr.detach(),
            "hr": hr.detach(),
            "sr": sr.detach()}
    
    def test_step(self, batch:Any, *args: Any, 
        **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        lr, hr = self._prep_batch(batch= batch)
        # sr = self.forward(lr)
        sr = stich_by_clip(lr, 4, self.generator)     
        return {
            "lr": lr.detach(),
            "hr": hr.detach(),
            "sr": sr.detach()}
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    
    def on_after_backward(self) -> None:
        _ = self._check_module_tensors(self.generator, 'after_backward', '')
    
    
    def _calc_loss(self, lr:torch.Tensor, hr:torch.Tensor, sr:torch.Tensor)->Dict:
        #elevation
        elev_loss = self.mse(sr, hr)
        with torch.no_grad():
        #slope
            slope_hr = self.slope_fn(hr)
            slope_sr = self.slope_fn(sr)
            slope_loss = self.mse(slope_sr, slope_hr)
        
        #river
            hr_m, sr_m = self._restore_real_values(hr, sr)
            hr_n = self._normalizer_min_max(hr_m)
            sr_n = self._normalizer_min_max(sr_m)
            hr_river = self.river_unet(hr_n)
            hr_river_heatmap = (torch.sigmoid(hr_river) > 0.5).float()
        
        river_loss, miou_0, miou_1, Miou = self.river_fn(sr_n, hr_river_heatmap, None)
        
        return { 'elev_loss': elev_loss,
                 'slope_loss': slope_loss,
                 'river_loss': river_loss,
                 'miou_0':miou_0,
                 'miou_1': miou_1,
                 "Miou": Miou    
        }
    
    def _unormalize(self, x:torch.Tensor) ->torch.Tensor:
        return x
    def _normalize(self, x:torch.Tensor)-> torch.Tensor:
        return x
    def _normalizer_min_max(self, x:torch.Tensor)-> torch.Tensor:
        """The Unet expects values between -1 e 1"""
        min = self.ram_normalizer.min.to('cuda')
        max = self.ram_normalizer.max.to('cuda')
        ran = (max - min).clamp_min(1e-6)
        return 2*(x -min)/ran - 1
        
        
    
    def _restore_real_values(self, hr:torch.Tensor, sr:torch.Tensor)->List[torch.Tensor]:
        # hr_01 = self._unormalize(hr)
        # sr_01 = self._unormalize(sr)
        batch = { 'hr_m': hr.clone(), 'sr_m': sr.clone()}
        self.ram_normalizer.revert(batch, 'sr_m')
        self.ram_normalizer.revert(batch, 'hr_m')
        return [batch['hr_m'], batch['sr_m']]
    

    
    def _filter_nan_batch(self, batch: Dict)-> Dict:
        """Check and Filter NaN values from the HR samples.

        Args:
            batch (Dict): Batch of samples

        Returns:
            Dict: Filtered batch of sample.
        """
        # batch = self._filter_outliers(batch) ao ativar isso o PSNR cai
        valid_indices = [i for i in range(batch['hr'].shape[0]) 
                        if torch.isfinite(batch['hr'][i]).all() and 
                        torch.isfinite(batch['lr'][i]).all()]
        return {
            'lr': batch['lr'][valid_indices],
            'hr': batch['hr'][valid_indices],
            'bounds': [batch['bounds'][i] for i in valid_indices],
            'crs': [batch['crs'][i] for i in valid_indices]}
    
    def _prep_batch(self, batch: Any)-> Tuple[torch.Tensor, torch.Tensor]:
        batch = self._filter_nan_batch(batch)                       
        # if utils2.check_hr_on_dict(batch):
        # normaliza LR no domínio do input e HR no domínio RAM (quando solicitado)
        self.input_normalizer(batch, 'lr')
        # if self.normalize_hr:                                    # type: ignore
        self.ram_normalizer(batch, 'hr')

        lr_img, hr_img = batch['lr'], batch['hr']
        # lr_img = self._normalize(lr_img)                        # type: ignore
        # if self.normalize_hr:                                   # type: ignore
        # hr_img = self._normalize(hr_img)                    # type: ignore

        return lr_img, hr_img

    def _tensor_stats(self, x: torch.Tensor, name: str)-> bool:
        """check if the gradient is exploding during training"""
        with torch.no_grad():
            if not torch.isfinite(x).all():
                mins = torch.nanquantile(x, 0).item() if torch.isfinite(x).any() else float('nan')
                maxs = torch.nanquantile(x,1).item() if torch.isfinite(x).any() else float('nan')
                print(f"[NF] {name}: finite={torch.isfinite(x).float().mean().item():.3f}, min={mins:.3e}, max={maxs:.3e}")
                return False
        return True

    def _check_module_tensors(self, module: torch.nn.Module, tag: str, prefix: str) -> bool:
        """Iterates through the parameters, grads, and buffers of a specific module (gen/disc)."""
        ok = True
        # parâmetros + gradientes
        for n, p in module.named_parameters(recurse=True):
            if p is None:
                continue
            ok &= self._tensor_stats(p,   f"{prefix}param[{tag}]::{n}")
            if p.grad is not None:
                ok &= self._tensor_stats(p.grad, f"{prefix}grad[{tag}]::{n}")
        # buffers (e.g., running_mean/var de BN)
        for n, b in module.named_buffers(recurse=True):
            if torch.is_tensor(b):
                ok &= self._tensor_stats(b, f"{prefix}buffer[{tag}]::{n}")
        return ok
