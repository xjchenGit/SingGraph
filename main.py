"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from dataloader import Dataset_SingFake, Dataset_SingFake_mert_w2v
from eval_metrics import compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from SingGraph import Wav2Vec2Model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(track,
                os.path.splitext(os.path.basename(args.config))[0],
                config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    # model = get_model(model_config, device)
    model = get_wav2vec2_model(model_config, device).to(device)

    # define dataloaders
    # trn_loader, dev_loader, eval_loader, additional_loader, persian_loader, mp3_loader, ogg_loader, aac_loader, opus_loader = get_singfake_loaders(args.seed, args, config)
    dataset_loaders = get_singfake_loaders(args.seed, args, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Evaluating on SingFake Eval Set...")
        
        eer_results = {} 
        for data_key in dataset_loaders:
            eer = evaluate(dataset_loaders[data_key], model, device)
            eer_results[data_key] = eer

        avg_codec_eer = (eer_results["codec_test/mp3_128k"] + 
                         eer_results["codec_test/ogg_64k"] +
                         eer_results["codec_test/adts_64k"] + 
                         eer_results["codec_test/opus_64k"]) / 4.
        
        print("Done. train_eer: {:.2f} %, test_set_eer: {:.2f} %, additional_test_eer: {:.2f} %, persian_eer: {:.2f} %".format(eer_results["train"] * 100,
                                                                                                                               eer_results["T01"] * 100,
                                                                                                                               eer_results["T02"] * 100,
                                                                                                                               eer_results["T04"] * 100))

        print("Codec testing: Average EER: {:.2f} %, MP3 EER: {:.2f} %, OGG EER: {:.2f} %, AAC EER: {:.2f} %, OPUS EER: {:.2f} %".format(avg_codec_eer * 100,
                                                                                                                                         eer_results["codec_test/mp3_128k"] * 100,
                                                                                                                                         eer_results["codec_test/ogg_64k"] * 100,
                                                                                                                                         eer_results["codec_test/adts_64k"] * 100,
                                                                                                                                         eer_results["codec_test/opus_64k"] * 100))
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(dataset_loaders["train"])
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(dataset_loaders["train"], model, optimizer, device,
                                   scheduler, config)

        dev_eer = evaluate(dataset_loaders["dev"], model, device)
        additional_eer = evaluate(dataset_loaders["T02"], model, device)

        print("DONE.\nLoss:{:.5f}, dev_eer: {:.2f} %, additional_test_eer: {:.2f} %".format(
                                            running_loss, dev_eer * 100, additional_eer * 100))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("additional_eer", additional_eer, epoch)
        
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                eval_eer = evaluate(dataset_loaders["T01"], model, device)
                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(), model_save_path / "best.pth")
                    
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(dataset_loaders["train"], model, device=device)
    eval_eer = evaluate(dataset_loaders["T01"], model, device)
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f} %".format(eval_eer * 100))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(), model_save_path / "best.pth")
        
    print("Exp FIN. EER: {:.3f} %".format(best_eval_eer * 100))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_wav2vec2_model(model_config: Dict, device: torch.device):
    model = Wav2Vec2Model(model_config, device)
    return model

def get_singfake_loaders(seed: int, args: argparse.Namespace, config: dict) -> List[torch.utils.data.DataLoader]:
    # base_dir = "../../dataset/split_dump_flac/"
    base_dir = "./dataset/split_dump_flac/"
    target_sr = float(config["target_sr"])
    
    # Define dataset keys and their corresponding paths
    dataset_keys = ["train", "dev", "T01", "T02", "T04",
                    "codec_test/mp3_128k", "codec_test/ogg_64k",
                    "codec_test/adts_64k", "codec_test/opus_64k"]
    datasets = {}

    # Common DataLoader settings
    common_settings = {
        "batch_size": config["batch_size"],
        "num_workers": 4,
        "pin_memory": True
    }

    # Initialize the generator for reproducibility
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Creating DataLoaders for each dataset
    for key in dataset_keys:
        dataset_path = os.path.join(base_dir, key)
        shuffle = True if key == "train" else False
        drop_last = True if key == "train" else False
        worker_init_fn = seed_worker if key == "train" else None
        generator = gen if key == "train" else None

        dataset = Dataset_SingFake_mert_w2v(args, config, base_dir=dataset_path, algo=args.algo,
                                            state="train" if key == "train" else "test", target_sr=target_sr)

        datasets[key] = DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, worker_init_fn=worker_init_fn,
                                   generator=generator, **common_settings)
    
    # Return DataLoaders in the specified order
    return datasets

def evaluate(loader, model, device: torch.device):
    """
    Evaluate the model on the given loader, then return EER.
    """
    model.eval()
    # we save target (1) scores to target_scores, and non target (0) scores to nontarget_scores.
    target_scores = []
    nontarget_scores = []
    debug = False
    count = 0
    with torch.no_grad():
        for batch_x, batch_x2, batch_y in tqdm(loader, total=len(loader)):
            batch_x, batch_x2 = batch_x.to(device), batch_x2.to(device)
            batch_out = model(batch_x, batch_x2)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            batch_y = batch_y.data.cpu().numpy().ravel()
            for i in range(len(batch_y)):
                if batch_y[i] == 1:
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
            count += 1
            if count == 10 and debug:
                break
    
    eer, _ = compute_eer(target_scores, nontarget_scores)
    return eer


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pbar = tqdm(trn_loader, total=len(trn_loader))
    for batch_x, batch_x2, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x, batch_x2 = batch_x.to(device), batch_x2.to(device) 
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_x2)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        pbar.set_description("loss: {:.5f}, running loss: {:.5f}".format(
            batch_loss.item(), running_loss / num_total))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu id to use (default: 0)")
    
    ##===================================================Rawboost data augmentation ===============================================================#

    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation =================================================================#
    
    main(parser.parse_args())
