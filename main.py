import torch
import torchvision.transforms.functional as F
from src.models.base import *

import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    
    model = EVFlowNet(args.train).to(device)
            
    if args.model.model_path:
        try:
            model_path = f"checkpoints/{args.model.model_name}.pth"
        except: # tryで例外が発生した場合
            print('model read Error')
        else: # tryで例外が発生しなかった場合
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"model: checkpoints/{args.model.model_name}.pth ")
    #else:
    #    model = EVFlowNet(args.train).to(device)
    
    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    # ------------------
    #   Start training
    # ------------------
    model.train()       # モデルをtrainモードにします

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    current_time = time.strftime("%Y%m%d%H%M%S")
        
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            # 元のソースコード
            # event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            # ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            # 複数受信
            total_photometric_loss = 0.
            flow_dict = model(event_image) # [B, 2, 480, 640]
            
            # for j in range(len(flow_dict)):
            #     for image_num in range(ground_truth_flow.shape[0]):
            #         flow = flow_dict["flow{}".format(j)][image_num]
            #         height = flow.shape[1]
            #         width = flow.shape[2]

            #         ground_truth_flow_resize = F.to_tensor(F.resize(F.to_pil_image(ground_truth_flow[image_num].cpu()),
            #                                                 [height, width])).cuda()
            #     loss: torch.Tensor = compute_epe_error(flow_dict["flow{}".format(j)], ground_truth_flow_resize)
            #     total_photometric_loss += loss  # .itemo() がいる？？ flowを変化させる？
            
            # -----------
            loss: torch.Tensor = compute_epe_error(flow_dict["flow{}".format(len(flow_dict)-1)], ground_truth_flow)
            total_photometric_loss += loss  # .itemo() がいる？？ flowを変化させる？
            # loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            
            
            print(f"batch {i} loss: {loss.item()}")
            print(f"batch {i} total_photometric_loss: {total_photometric_loss.item()}")
            optimizer.zero_grad()
            total_photometric_loss.backward()
            # loss.backward()
            optimizer.step()

            total_loss += total_photometric_loss.item()
            #total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')
        
        model_path = f"checkpoints/model_{current_time}_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}_epoch{epoch}_finish.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()       # モデルをevaluationモードにします
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow_dict = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow_dict['flow3']), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
