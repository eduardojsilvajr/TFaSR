from torchgeo.datasets import RasterDataset, VectorDataset, BoundingBox, GeoDataset
from torchgeo.samplers import GridGeoSampler
from shapely.geometry import mapping
import torch, os, glob
import rasterio
import rasterio.features
from rasterio.transform import rowcol
from rtree import index as rtree_index
from affine import Affine
import geopandas as gpd
from typing import Any, Dict, Optional, List
import numpy as np
from torchgeo.samplers.constants import Units

# Classes
class DEMDataset(RasterDataset):
    """Dataset padrão (file-backed) para 1 banda, vários GeoTIFFs."""
    is_image = True
    mint = 0
    maxt = 0
    def __init__(self, paths: List[str]):
        if not paths:
            raise FileNotFoundError("Nenhum GeoTIFF encontrado.")
        super().__init__(paths=paths)  # TorchGeo cria o índice espacial e mosaica sob demanda
        
        # NaN Limit
        self.threshold = -10
    
    def __getitem__(self, query: "BoundingBox") -> Dict[str, Any]:
        out = super().__getitem__(query)
        # creating NaN mask
        img = out["image"].to(torch.float32)     
        img[img < self.threshold] = float("nan") 
        out["image"] = img
        return out      
    

class InMemoryGeoRaster(RasterDataset):
    """
    Subclasse de RasterDataset que carrega todo o GeoTIFF em memória,
    para que __getitem__(query) retorne patches sem abrir o arquivo a cada vez.
    """
    is_image = True

    def __init__(self, path: str):
        # Chama o __init__ do RasterDataset (configura CRS, bounds etc.)
        super().__init__(paths=path)

        # Abre apenas UMA vez e armazena o array e transform
        with rasterio.open(path) as src:
            self._array     = src.read(1)          # numpy array de shape (H, W)
            self._transform = src.transform        # Affine transform
            self.crs        = src.crs              # necessário para GeoDataset
            self.bbox     = src.bounds           # (minx, miny, maxx, maxy)
        self._height, self._width = self._array.shape
        
        # realçar os valores NaN
        # self._array[self._array<0] = float('nan')       
     
    def __len__(self):
        return 1
    
    def _put_nan_values(self):
        return NotImplemented
    @property
    def bounds(self) -> BoundingBox:
        # agora retorna um BoundingBox com tempo definido (mint=0, maxt=0)
        left, bottom, right, top = self.bbox
        return BoundingBox(
            minx=left, maxx=right,
            miny=bottom, maxy=top,
            mint=0,    maxt=0
        )
        
    def __getitem__(self, query: BoundingBox) -> dict[str, torch.Tensor]:
        """
        Recebe um BoundingBox (minx, miny, maxx, maxy, mint, maxt),
        converte para índices de pixel e fatia self._array na RAM.
        """

        # 1) Extrai limites em coordenadas de mundo
        try:
            minx, miny, maxx, maxy = query.left, query.bottom, query.right, query.top
        except:
            minx, miny, maxx, maxy = query.minx, query.miny, query.maxx, query.maxy

        # 2) Converte (x, y) → (row, col)
        row0, col0 = rowcol(self._transform, minx, maxy)  # canto superior-esq
        row1, col1 = rowcol(self._transform, maxx, miny)  # canto inferior-dir

        # 3) Ordena índices para garantir slice correto
        row_start, row_stop = sorted((row0, row1))
        col_start, col_stop = sorted((col0, col1))

        # 4) Fatia o array em memória: shape resultante (h, w)
        patch_np = self._array[row_start:row_stop, col_start:col_stop]
        

        # 5) Converte para Tensor e adiciona canal único: (1, h, w)
        patch_t = torch.from_numpy(patch_np).unsqueeze(0).float()

        return {
            "image": patch_t,
            "crs":   self.crs,
            "bounds": query,
        }
    

class CustomDEMDataset(GeoDataset):
    def __init__(self, dataset, ram_ds):
        super().__init__(transforms=None)
        # composto em 30 m
        self.dataset = dataset 
        
        self.hr_ds = ram_ds           
        # grade espacial total (CRS + bounds) vinda do LR
        self.crs = dataset.crs     
        self.bbox = dataset.bounds
        self.index = dataset.index
        self.res = dataset.res  
        
    @property
    def bounds(self):
        return self.bbox  

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        # LR já veio em 30 m sem rampear o próprio ram_ds
        lr_patch = self.dataset[query]["image"]
        # HR em 15 m, diretamente do ram_ds
        if self.hr_ds:
            hr_patch = self.hr_ds[query]["image"]
            return {
                "lr": lr_patch,
                "hr": hr_patch,
                "bounds": query,
                "crs": self.crs,
            }
        else:
            return {
                "lr": lr_patch,
                "bounds": query,
                "crs": self.crs,
            }
            
class VectorDataset_GDF(VectorDataset):
    def __init__(self, path, layer, res=0.0001,  crs=None):
        GeoDataset.__init__(self, transforms=None)
        self.gdf = gpd.read_file(path, layer=layer)
        self.crs = crs or self.gdf.crs
        self.res = res

        # monta um R-tree 3D (x, y, t)
        prop = rtree_index.Property()
        prop.dimension = 3
        self.index = rtree_index.Index(properties=prop)

        for i, geom in enumerate(self.gdf.geometry):
            minx, miny, maxx, maxy = geom.bounds
            # para t usamos 0…0
            coords3d = (minx, miny, 0, maxx, maxy, 0)
            self.index.insert(i, coords3d, obj=i)
            
    @property
    def bounds(self) -> BoundingBox:
        minx, miny, maxx, maxy = self.gdf.total_bounds
        # mint e maxt são temporais. Como não estamos usando tempo, definimos como 0
        return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=0)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        # recupera índices das geometrias que intersectam a janela
        hits = list(self.index.intersection((query.minx, query.miny,
                                              query.maxx, query.maxy,
                                              query.mint, query.maxt),
                                             objects=True))
        height = round((query.maxy - query.miny) / self.res)
        width  = round((query.maxx - query.minx) / self.res)
        if not hits:
            # retorna máscara vazia

            mask = torch.zeros((1, height, width), dtype=torch.uint8)
            mask_id = torch.zeros((1, height, width), dtype=torch.uint8)
            return {'aoi': mask, 'aoi_id': mask_id, 'crs': self.crs, 'bounds': query}

        shapes, shapes_id = [], []
        for hit in hits:
            idx = hit.object
            geom = self.gdf.geometry.iloc[idx]
            rid  = self.gdf.geometry.apply(lambda g: g.equals(geom)).idxmax() #int(self.gdf.iloc[idx]["fid"]) 
            gj   = mapping(geom)
            shapes.append((gj, 1))
            shapes_id.append((gj, rid+1))

        # rasterização
        
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy,
            width, height
        )
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        mask_id = rasterio.features.rasterize(
            shapes_id,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        return {'aoi': torch.from_numpy(mask).unsqueeze(0), 
                'aoi_id': torch.from_numpy(mask_id).unsqueeze(0),
                'crs': self.crs, 'bounds': query}

class StrictGridGeoSampler(GridGeoSampler):
    """
    Gera janelas do GridGeoSampler e filtra garantindo que cada bbox
    esteja TOTALMENTE contida dentro de alguma região válida da AOI.

    Se `cover` for None → cai no teste simples de conter dentro do bounds do dataset.
    Se `cover` for um GeoDataset que retorna 'aoi' (máscara binária) → aceita somente
    janelas com máscara totalmente 1 (cobertura integral).
    """
    def __init__(self, dataset, size, stride, roi, units: Units = Units.PIXELS, cover: Optional[object] = None):
        super().__init__(dataset=dataset, size=size, stride=stride, roi=roi, units=units)
        self._dataset = dataset 
        self.cover = cover  # ex.: seu VectorDataset_GDF de teste
        
        self.res = getattr(dataset, "res", None)  # p.ex. 0.00026949...
        if self.res is None:
            raise ValueError("Dataset precisa expor `res` (mundo por pixel) para ancoragem por AOI.")
        self.res_x = float(self.res)
        self.res_y = float(self.res) 

    def _inside_dataset_bounds(self, bbox):
        ds_bounds = self._dataset.bounds  # não use self.index.bounds aqui
        return (
            bbox.minx >= ds_bounds.minx and bbox.maxx <= ds_bounds.maxx and
            bbox.miny >= ds_bounds.miny and bbox.maxy <= ds_bounds.maxy
        )

    def _fully_covered_by_aoi(self, bbox):
        out = self.cover[bbox]  # rasteriza AOI na janela (resolve multi-região corretamente)
        mask = out['aoi']
        mid = out['aoi_id']
        # aceita somente se todos os pixels estiverem dentro da AOI
        if mask.numel() == 0 or not torch.all(mask == 1):
            return False

        uniq = torch.unique(mid[mid > 0])
        return uniq.numel() == 1

    def _iter_grid_over_box(self, minx, miny, maxx, maxy):
        """Gera bboxes de tamanho `size` ancorando em (minx,maxy)."""
        w_world = self.size[0] #/ self.res_x
        h_world = self.size[1] #/ abs(self.res_y)

        # começa exatamente no canto superior-esquerdo da AOI
        y_top = maxy
        while (y_top - h_world) >= miny - 1e-12:
            x_left = minx
            while (x_left + w_world) <= maxx + 1e-12:
                yield BoundingBox(
                    minx=x_left,
                    maxx=x_left + w_world,
                    miny=y_top - h_world,
                    maxy=y_top,
                    mint=0, maxt=0
                )
                x_left += self.stride[0] #* self.res_x
            y_top -= self.stride[1] #* abs(self.res_y)
    
    def __iter__(self):
        # Se não houver AOI → comportamento original
        if self.cover is None or not hasattr(self.cover, "gdf"):
            for bbox in super().__iter__():
                if self._inside_dataset_bounds(bbox):
                    yield bbox
            return

        # Garante que a rasterização da AOI use a MESMA resolução do raster
        # (evita “falsos negativos” nas bordas)
        if hasattr(self.cover, "res"):
            self.cover.res = self.res  # alinhar resolução da máscara ao raster

        # Para cada feição da AOI, gera grade a partir do seu canto sup-esq
        for geom in self.cover.gdf.geometry:
            minx, miny, maxx, maxy = geom.bounds
            for bbox in self._iter_grid_over_box(minx, miny, maxx, maxy):
                if self._inside_dataset_bounds(bbox) and self._fully_covered_by_aoi(bbox):
                    yield bbox
    
    
    
    # def __iter__(self):
    #     for bbox in super().__iter__():
    #         if self.cover is None:
    #             if self._inside_dataset_bounds(bbox):
    #                 yield bbox
    #         else:
    #             # testa bounds globais primeiro (barato), depois cobertura integral por AOI (preciso)
    #             if self._inside_dataset_bounds(bbox) and self._fully_covered_by_aoi(bbox):
    #                 yield bbox
    
    