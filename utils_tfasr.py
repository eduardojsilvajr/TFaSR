import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchgeo.datasets.utils import BoundingBox
from typing import List

import random, os, glob, math
import torch



def inicialize_normalizer_by_tiles(
    ds,
    mode: str = "min_max",             # 'min_max' ou 'mean_std'
    roi: BoundingBox | None = None,    # por padrão usa ds.bounds
    nodata: float | None = None,       # valor NoData (ex.: -32768); None => só filtra não-finito
    max_tiles: int | None = None,      # limite opcional de tiles para acelerar
):
    """
    Calcula estatísticas varrendo *cada tile* retornado pelo índice espacial do RasterDataset.
    Evita GridGeoSampler e qualquer canvas grande.

    Retorna:
      - utils2.Normalizer_minmax(...) se mode='min_max'
      - utils2.Normalizer(...)       se mode='mean_std'
    """

    roi = roi or ds.bounds
    idx = ds.index
    dim = getattr(idx.properties, "dimension", 2)

    # Monta tupla de consulta compatível com a dimensão do índice (2D ou 3D)
    if dim == 3:
        query = (roi.minx, roi.miny, roi.maxx, roi.maxy, roi.mint, roi.maxt)
    else:
        query = (roi.minx, roi.miny, roi.maxx, roi.maxy)

    # Coleta itens (tiles) que intersectam o ROI
    items = list(idx.intersection(query, objects=True))
    if len(items) == 0:
        raise RuntimeError("Nenhum tile encontrado no ROI ao consultar o índice espacial.")

    # Acumuladores globais
    has_data = False
    g_min = torch.tensor(float("inf"))
    g_max = torch.tensor(float("-inf"))

    # Itera tile a tile
    for i, it in enumerate(items):
        if max_tiles is not None and i >= max_tiles:
            break

        # BBox do tile (respeita dimensão do índice)
        bb = it.bounds  # rtree IndexItem: tuple com 4 ou 6 números
        if dim == 3:
            minx, maxx, miny, maxy, mint, maxt = bb
        else:
            minx, maxx, miny, maxy = bb
            mint, maxt = ds.bbox.mint, ds.bbox.maxt

        tile_bb = BoundingBox(minx, maxx, miny, maxy, 0, 0)

        # Interseção com o ROI (segurança)
        ix_minx = max(tile_bb.minx, roi.minx)
        ix_maxx = min(tile_bb.maxx, roi.maxx)
        ix_miny = max(tile_bb.miny, roi.miny)
        ix_maxy = min(tile_bb.maxy, roi.maxy)
        if not (ix_minx < ix_maxx and ix_miny < ix_maxy):
            continue  # sem sobreposição espacial

        inter = BoundingBox(ix_minx, ix_maxx, ix_miny, ix_maxy, mint, maxt)

        # Lê apenas a área de interseção via RasterDataset (abrirá o GeoTIFF certo)
        out = ds[inter]               # {'image': Tensor [1,H,W], ...}
        x = out["image"].to(torch.float32)
        x[x<0] = float('nan')

        # Máscara de inválidos
        if nodata is not None:
            invalid = torch.isclose(x, torch.tensor(float(nodata), dtype=x.dtype))
        else:
            invalid = ~torch.isfinite(x)

        if invalid.all():
            continue

        # Atualiza min/máx ignorando inválidos
        xv = x.masked_fill(invalid, float('nan'))
        
        flat = x.view(1,-1)
        mask = torch.isnan(flat)

        filled_min = flat.clone()
        filled_min[mask] = float('inf')
        bmin,_ = filled_min.min(dim=1)

        filled_max = flat.clone()
        filled_max[mask] = float('-inf')
        bmax, _ = filled_max.max(dim=1)
        
        if torch.isfinite(bmin):
            g_min = torch.minimum(g_min, bmin)
            has_data = True
        if torch.isfinite(bmax):
            g_max = torch.maximum(g_max, bmax)
            has_data = True

    if not has_data:
        raise RuntimeError("Apenas NoData/NaN encontrados ao percorrer os tiles do ROI.")

    # Constrói o normalizador
    
    g_min = g_min
    g_max = g_max
    return Normalizer_minmax(min=g_min, max=g_max)

def lista_tifs(diretorio: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(diretorio, pattern)))

class Normalizer_minmax(nn.Module):
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        super().__init__()

        self.min = min[:, None, None]
        self.max = max[:, None, None]
        self.range = (self.max - self.min).clamp_min(1e-6)
        
    def forward(self, inputs: dict,label: str):
        x = inputs[label][..., : len(self.min), :, :]
        min_   = self.min.to(x.device)
        range_ = self.range.to(x.device)
        if inputs[label].ndim == 4:
            x = 2*(x - min_[None, ...]) / range_[None, ...] -1
        else:
            x = 2*(x - min_[:x.shape[0]]) / range_[:x.shape[0]] -1

        inputs[label][..., : len(self.min), :, :] = x
    
        return inputs

    def revert(self, inputs: dict, label: str):
        """
        De-normalize the batch.
        Args:
            inputs (dict): Dictionary with the 'label' key
        """
        x = inputs[label][..., : len(self.min), :, :]
        min_   = self.min.to(x.device)
        range_ = self.range.to(x.device)
                
        if x.ndim == 4:
            x = .5*(1+x) * range_[None, ...] + min_[None, ...]
            # x = self.range[None, ...]**x + self.min[None, ...]
        else:
            x = .5*(1+x) * range_[:x.shape[0]] + min_[:x.shape[0]]
            # x = self.range[:x.shape[0]]**x + self.min[:x.shape[0]]

        inputs[label] = x

        return inputs

def set_seed(seed: int):
    """Configura a semente para reprodutibilidade."""
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # para usar o pad 'reflect', tem comentar aqui
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    
    
def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            # assume metadados como crs, bbox etc., só replica o primeiro
            # collated[key] = batch[0][key]
            collated[key] = [sample[key] for sample in batch]
    # tensor of size [A,1,patch_size,patch_size]
    # collated['image'] = collated['image'].reshape(-1,patch_size,patch_size).unsqueeze(1)
    return collated


def stich_by_clip(img: torch.Tensor, 
    n: int, generator: nn.Module, 
    pad: int = 15, scale: int=4 
) -> torch.Tensor:
    
    b, c, H, W = img.shape

    # aplica padding reflexivo uma única vez
    img_pad = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')

    # calcula tamanho aproximado de cada patch na região original
    ph = math.ceil(H / n) #256
    pw = math.ceil(W / n) # 256

    # prepara tensor de saída
    H_sr, W_sr = H * scale, W * scale
    out = torch.zeros((b, c, H_sr, W_sr), device=img.device)

    cp = pad * scale  # quantidade a recortar em SR

    with torch.no_grad():
        for i in range(n):
            y0 = i * ph
            y1 = min((i + 1) * ph, H)
            ys, ye = y0 + pad, y1 + pad

            for j in range(n):
                x0 = j * pw
                x1 = min((j + 1) * pw, W)
                xs, xe = x0 + pad, x1 + pad

                # extrai patch LR + contexto de 'pad' px em volta
                lr_patch = img_pad[:, :, ys - pad : ye + pad,
                                       xs - pad : xe + pad]

                # gera SR para o patch completo
                bic_patch = F.interpolate(lr_patch, scale_factor=4, mode='bicubic' )
                sr_patch = generator(bic_patch)

                # recorta o contexto (pad*scale) de cada borda
                sr_center = sr_patch[
                    :,
                    :,
                    cp : sr_patch.size(2) - cp,
                    cp : sr_patch.size(3) - cp
                ]

                # cola o patch central no lugar correto da saída
                out[
                    :,
                    :,
                    y0 * scale : y1 * scale,
                    x0 * scale : x1 * scale
                ] = sr_center
    return out