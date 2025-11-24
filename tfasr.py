import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" 
os.environ["PYTHONHASHSEED"] = "42"
os.environ['CUDA_LAUNCH_BLOCKING']='1'
PROJECT_ROOT = "D:\\Documentos\\OneDrive\\Documentos\\Mestrado\\SR - MDE"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRIU_DIR = os.path.join(PROJECT_ROOT, "SRIU")
if SRIU_DIR not in sys.path:
    sys.path.insert(0, SRIU_DIR)
import torch
from SRIU.torchgeo_modules import DEMDataset, CustomDEMDataset, VectorDataset_GDF, StrictGridGeoSampler
from SRIU.L_callbacks import Compute_metrics, Plot_results, Save_Gtiff
from tfasr_model import Generator as srnet
from FeatureLoss import RiverLoss
from DEM_features import Slope
from utils_tfasr import *
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from RiverModel.unet.unet_model import UNet as RiverNet
from L_module import TfarSR

import lightning as L
import mlflow

dem_type = 'dsm'

#controle de determinismo
set_seed(42)
seed = torch.Generator().manual_seed(42)
L.seed_everything(42, workers=True)

mlflow_logger = MLFlowLogger(
        experiment_name="TfaSR",                 # nome do experimento,
        run_name='TfaSR_teste',
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
torch.set_float32_matmul_precision('high')

#criando os datasets
input_directory = f"C:\\Users\\Eduardo JR\\Fast\\SRIU\\zoom_4"
input_train = DEMDataset(paths=lista_tifs(input_directory+'/train', 'cop*.tif'))
input_val = DEMDataset(paths=lista_tifs(input_directory+'/val', 'cop*.tif'))
input_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'cop*.tif'))


ram_train = DEMDataset(paths=lista_tifs(input_directory+'/train', f'ram_*_{dem_type}.tif'))
ram_val = DEMDataset(paths=lista_tifs(input_directory+'/val', f'ram_*_{dem_type}.tif'))
ram_test = DEMDataset(paths=lista_tifs(input_directory+'/test', f'ram_*_{dem_type}.tif'))

train_aoi = VectorDataset_GDF(
        path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
        layer='esrgan_finetuning_50k_train', 
        crs="EPSG:4326")
    
val_aoi = VectorDataset_GDF(
        path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
        layer='esrgan_finetuning_50k_val', 
        crs="EPSG:4326")
test_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetuning_50k_test', 
                crs="EPSG:4326")

train_ds =  CustomDEMDataset(input_train,ram_train)
val_ds = CustomDEMDataset(input_val, ram_val)
test_ds = CustomDEMDataset(input_test, ram_test)

train_sampler = RandomGeoSampler(
                train_ds,
                size=128,
                roi=train_aoi.bounds)
    
train_dataload = DataLoader(dataset=train_ds,
                            batch_size=6,
                            sampler=train_sampler,
                            generator= seed,
                            pin_memory=False,
                            num_workers=0,
                            collate_fn= custom_collate_fn)

val_sampler = StrictGridGeoSampler(
                dataset=val_ds,
                size=128,
                stride=128,
                roi=val_ds.bounds,
                cover= val_aoi) 
    
val_dataload = DataLoader(dataset=val_ds, 
                            batch_size=6, 
                            sampler=val_sampler,
                            generator= seed,
                            pin_memory=False,
                            num_workers=0,   
                            collate_fn=custom_collate_fn)

test_sampler = StrictGridGeoSampler(
                dataset=test_ds,
                size=910,   #256
                stride=910, #218
                roi=test_ds.bounds,
                cover= test_aoi)
    
test_dataload = DataLoader(dataset=test_ds, 
                        batch_size=1, 
                        sampler=test_sampler,
                        generator= seed,
                        pin_memory=False,
                        #   persistent_workers=True,
                        num_workers= 0,
                        collate_fn=custom_collate_fn)

#normalizadores
input_norm = inicialize_normalizer_by_tiles(input_train, mode='min_max', nodata=None)
ram_norm = inicialize_normalizer_by_tiles(ram_train, mode='min_max', nodata=-9999 )


#Ligthining Callbacks
early_stopping = EarlyStopping(
                monitor="v_loss",
                patience=10,
                mode="min",
                verbose=True
)
compute_metrics = Compute_metrics(ram_norm=ram_norm, input_norm=ram_norm, scale=4)

plot_results = Plot_results(val_batch_interval=40, test_batch_interval=3, 
                                ram_norm=ram_norm, input_norm=ram_norm, cmap='terrain', 
                                dir_path=r'D:\Documentos\OneDrive\Documentos\Mestrado\SR - MDE\TFaSR_code\artifacts')


save_ckpt = ModelCheckpoint(
            monitor="v_loss",
            save_top_k=1,
            mode="min",
            filename= 'TfaSR_e_{epoch:02d}',
            dirpath="tfasr_checkpoint/",
            auto_insert_metric_name=False
)

#lightining trainer

trainer = L.Trainer(max_epochs=100, 
                precision="16-mixed", # "32-true" ,
                accelerator="gpu",
                deterministic= True,
                limit_train_batches=0.10,
                # limit_val_batches=.50,
                # fast_dev_run=2,
                logger=mlflow_logger,
                enable_progress_bar=True,
                # profiler='simple',
                # limit_predict_batches=.5,  
                callbacks=[save_ckpt, early_stopping, compute_metrics, plot_results],
)

modelo = srnet(n_residual_blocks=16, upsample_factor=4)
modelo.load_state_dict(torch.load(r'tfasr_checkpoint\tfasr_099.pth', map_location='cpu'))

slope_fn = Slope()

river_unet = RiverNet(n_channels=1, n_classes=1)
extract_river_Weights = r'.\RiverModel\unet\unet.pth'
river_unet.load_state_dict(torch.load(extract_river_Weights, map_location='cpu'))
river_criterion = RiverLoss(unet=river_unet)


tfasr = TfarSR(generator=modelo, hr_norm=ram_norm,
               lr_norm=input_norm, learning_rate=1e-4,
               slope_fn=slope_fn, river_fn=river_criterion, 
               river_unet=river_unet)

# tfasr = TfarSR.load_from_checkpoint(checkpoint_path='tfasr_checkpoint/TfaSR_e_24.ckpt', map_location='cpu', 
#                 generator=modelo, hr_norm=ram_norm,
#                 lr_norm=input_norm,learning_rate=1e-4,
#                 slope_fn=slope_fn, river_fn=river_criterion, 
#                 river_unet=river_unet )

trainer.fit(model=tfasr, train_dataloaders=train_dataload, val_dataloaders=val_dataload)
trainer.test(model=tfasr, dataloaders=test_dataload, ckpt_path='best')

with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_artifacts("TFaSR_code/artifacts/test", artifact_path="test_imgs")
        mlflow.log_artifact(save_ckpt.best_model_path, artifact_path='model')
        # mlflow.log_artifact('tfasr_checkpoint/TfaSR_e_24.ckpt', artifact_path='model')


