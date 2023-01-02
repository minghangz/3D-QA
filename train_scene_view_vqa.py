if __name__ == "__main__":
    START_METHOD = "forkserver"
    import multiprocessing

    multiprocessing.set_start_method(START_METHOD, force=True)
    import torch
    import torch.multiprocessing
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    torch.multiprocessing.set_start_method(START_METHOD, force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(torch.multiprocessing.get_start_method())
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    import pretty_errors
    import os, sys, json, glob, logging, random, pickle, warnings, colorama, datasets, toml, transformers
    from icecream import ic
    from collections import defaultdict, Counter
    from datetime import datetime
    from pprint import pprint
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        load_dataset,
        load_metric,
    )
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    import itertools
    import numpy as np
    from typing import Dict, List, Any, Optional, Union, Set

    sys.path.append(".")
    # sys.path.append("../bert-vqa")
    from multiprocessing.spawn import freeze_support
    from utils_eval_blip import *
    from copy import copy, deepcopy

    freeze_support()
    seed_all(42)

    DSET_PATH = {
        "test_w_obj": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_test_w_obj.json",
        "test_wo_obj": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_test_wo_obj.json",
        "train": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_train.json",
        "val": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_val.json",
    }
    DSET_VIEWS_PATH = "/home/mowentao/data/ScanQA/data/rendered_images_new"
    SCAN_NAMES = list(
        filter(
            lambda n: n.endswith("00"),
            sorted(
                [
                    line.rstrip()
                    for line in open(
                        "/home/mowentao/data/ScanQA/data/scannet/meta_data/scannetv2.txt"
                    )
                ]
            ),
        )
    )
    SCENE_FEAT_PATH = "scene_blip_features.pkl"

    # SCAN_NAMES = SCAN_NAMES[:100]

    torch.backends.cudnn.benchmark = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = (
        colorama.Fore.MAGENTA
        + "[%(asctime)s %(name)s %(levelname)s] "
        + colorama.Fore.WHITE
        + "%(message)s"
    )

    def parse_args():
        from argparse import ArgumentParser

        parser = ArgumentParser()
        # --- OPTIONS BEGIN ---
        parser.add_argument("--topk", type=int, default=3)
        parser.add_argument("--i2tfile", type=str, default="scene_eval.json")
        parser.add_argument("--dset_views_path", type=str, default=DSET_VIEWS_PATH)
        parser.add_argument("--split", type=str, default="all")
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--topk_images", type=int, default=1)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=64)
        parser.add_argument("--scene_range", type=str, default=":")
        parser.add_argument("--coeff_selector", type=float, default=1)
        parser.add_argument("--use_selector", action="store_true")
        # --- OPTIONS END ---
        return parser.parse_args()

    args = parse_args()

    DSET_VIEWS_PATH = args.dset_views_path

    beg, end = list(
        map(lambda s: None if s == "" else int(s), args.scene_range.split(":"))
    )
    SCAN_NAMES = SCAN_NAMES[beg:end]

    if args.split == "all":
        args.split = ["train", "val", "test_w_obj", "test_wo_obj"]
    else:
        args.split = args.split.strip().split(",")

    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        local_rank = dist.get_rank()
    else:
        local_rank = 0

    if local_rank == 0:
        logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")
    else:
        logging.basicConfig(
            format=log_format, level=logging.CRITICAL, datefmt="%I:%M:%S"
        )

    logger.info(f"--- OPTIONS BEGIN ---")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"--- OPTIONS END ---")

    device = torch.device(f"cuda:{0}")
    torch.cuda.set_device(device)

    # --- Load scanqa dset
    def load_dset(dset_path):
        import json

        logging.info("Loading Dataset...")
        dset_dict = {}
        for split, p in dset_path.items():
            data = json.load(open(p, "r"))
            dset = datasets.Dataset.from_dict(
                {k: [s[k] for s in data] for k in data[0].keys()}
            )
            dset_dict[split] = dset
        return datasets.DatasetDict(dset_dict)

    dset = load_dset(DSET_PATH)
    logger.info(dset)

    all_answers = []
    all_scans = []

    def cnt(s):
        global all_answers
        all_answers += sum(s["answers"], start=[])

    def cnt_scans(s):
        global all_scans
        all_scans += s["scene_id"]

    for split in ["train", "val"]:
        dset[split].map(cnt, batched=True)
    all_answers = Counter(all_answers)
    all_answers = sorted([a for a, n in all_answers.items() if n >= 5])
    logger.info(f"Total {len(all_answers)} answers.")

    for split in dset.keys():
        dset[split].map(cnt_scans, batched=True)

    SCAN_NAMES = sorted(set(all_scans).intersection(SCAN_NAMES))
    logger.info(f"Total {len(SCAN_NAMES)} scenes.")

    feature_exists = os.path.exists(SCENE_FEAT_PATH)
    if True or not feature_exists:
        pool = SceneViewsPool(DSET_VIEWS_PATH, SCAN_NAMES, preprocess=preprocess_vqa)
        images = pool.images

    logger.info(f"Loaded images.")

    # --- Init BLIP Model
    # from models.blip import blip_decoder
    # from models.blip_pretrain import blip_pretrain
    # from models.blip_itm import blip_itm
    from models.blip_vqa import blip_vqa

    logger.info("Loading BLIP Models...")

    model_vqa = blip_vqa(
        pretrained="ckpts/model_base_vqa_capfilt_large.pth",
        image_size=480,
        vit="base",
        use_selector=args.use_selector,
        coeff_selector=args.coeff_selector,
    )
    model_vqa = model_vqa.to(device)

    if world_size > 1:
        # Use multi-gpu
        model_vqa = torch.nn.parallel.DistributedDataParallel(
            model_vqa, find_unused_parameters=True
        )
        model_inner = model_vqa.module
    else:
        model_inner = model_vqa

    optim_params = [
        {"params": [param for name, param in model_vqa.named_parameters() if name.find("selector") != -1], "lr": 1e-4}, 
        {"params": [param for name, param in model_vqa.named_parameters() if name.find("selector") == -1], "lr": 1e-5}, 
    ]
    optimizer = torch.optim.AdamW(optim_params)

    # --- Train
    dset_val = datasets.concatenate_datasets([dset[split] for split in ["val"]])
    dset = datasets.concatenate_datasets([dset[split] for split in ["train"]])
    
    dset = dset.filter(lambda s: s["scene_id"] in SCAN_NAMES)
    dset_val = dset_val.filter(lambda s: s["scene_id"] in SCAN_NAMES)

    logger.info(f"Total {dset.num_rows} training samples.")
    logger.info(f"Total {dset_val.num_rows} validation samples.")
    logger.info(f"Training on x{args.topk_images} images. [topk-training]")

    tmp = json.load(open(args.i2tfile, "r"))
    pred = tmp["view"]
    if world_size == 1:
        dataloader_train = DataLoader(
            dset,
            batch_size=args.train_batch_size // args.topk_images,
            collate_fn=collate_features_simple,
            shuffle=True,
        )
        dataloader_val = DataLoader(
            dset_val,
            batch_size=args.eval_batch_size // args.topk_images,
            collate_fn=collate_features_simple,
        )
    else:
        dataloader_train = get_ddp_dataloader(
            dset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_features_simple,
        )
        dataloader_val = get_ddp_dataloader(
            dset_val,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_features_simple,
        )

    for eid in range(args.epochs):
        K = args.topk_images
        logger.info(f"[{eid}]start training...")
        model_vqa.train()
        for batch in dataloader_train:
            questions = batch["question"]
            batch_size = len(questions)
            question_ids = batch["question_id"]
            scene_names = batch["scene_id"]
            gt_answers = batch["answers"]

            questions = questions * K
            scene_names = scene_names * K
            gt_answers = gt_answers * K

            image_names = sum(
                [
                    [pred[question_id][i] for question_id in question_ids]
                    for i in range(K)
                ],
                start=[],
            )
            image_batch = torch.stack(
                [
                    images[scene_name][image_name]
                    for scene_name, image_name in zip(scene_names, image_names)
                ]
            ).to(device)
            gt_answer_for_training = [
                random.choice(gt_answer_list) for gt_answer_list in gt_answers
            ]
            loss, loss_selector = model_vqa(
                image_batch,
                questions,
                train=True,
                answer=gt_answer_for_training,
                n=[1] * batch_size * K,
                weights=torch.ones(batch_size * K, device=device),
            )
            if loss_selector is not None:
                logger.info(f"{loss_selector.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

        logger.info(f"[{eid}]start validation...")
        total, correct = 0, 0
        model_vqa.eval()
        with torch.no_grad():
            for batch in dataloader_val:
                # K = args.topk_images
                K = 1
                questions = batch["question"]
                question_ids = batch["question_id"]
                scene_names = batch["scene_id"]
                gt_answers = batch["answers"]
                batch_size = len(questions)

                image_names = sum(
                    [
                        [pred[question_id][i] for question_id in question_ids]
                        for i in range(K)
                    ],
                    start=[],
                )
                if K > 1:
                    questions = questions * K  # => B*K
                    scene_names = scene_names * K
                    gt_answers = gt_answers * K
                image_batch = torch.stack(
                    [
                        images[scene_name][image_name]
                        for scene_name, image_name in zip(scene_names, image_names)
                    ]
                ).to(device)

                ans_idx, ans_score = model_vqa(
                    image_batch,
                    questions,
                    train=False,
                    inference="rank",
                    answer=all_answers,
                    k_test=128,
                )

                sorted_idx = ans_score.argsort(dim=-1, descending=True)

                all_answer_score = torch.zeros(
                    [batch_size, len(all_answers)], device=device
                )

                # ic(all_answer_score.shape, ans_idx.shape, ans_score.shape)
                for i in range(ans_score.size(0)):
                    all_answer_score[i % batch_size][ans_idx[i]] += ans_score[i]
                all_answer_score = torch.where(
                    all_answer_score == 0, -1e6, all_answer_score
                )
                max_ids = all_answer_score.argmax(dim=-1)

                answers = [all_answers[max_ids[i].item()] for i in range(batch_size)]

                # answers = [
                #     all_answers[ans_idx[i][sorted_idx[i]][0].item()] for i in range(ans_idx.size(0))
                # ]

                total += len(gt_answers)
                correct += sum(
                    [
                        1 if answers[i] in gt_answer else 0
                        for i, gt_answer in enumerate(gt_answers)
                    ]
                )

        if world_size > 1:
            total = sum(all_gather_concat_list([total], world_size))
            correct = sum(all_gather_concat_list([correct], world_size))

        logging.info(f"[{eid}]acc@1 {correct / total * 100:.2f}")
        torch.cuda.empty_cache()
    # --- Save prediction
    # json.dump({"view": pred, "answer": pred_answer}, open(args.outfile, "w"))

    logging.info("Finished Training")
