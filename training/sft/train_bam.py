"""
Entry point for BAM (Residual Hidden Adapter) training.

Trains a single BAM model for a single dataset at a time.
  --task_type cls  →  BAMCLS model (classification loss)
  --task_type qa   →  BAMQA  model (LM teacher-forcing loss)

The shell script (train_bam.sh) is responsible for sweeping across datasets and
calling this script once per dataset, passing the appropriate --task_type and
per-dataset JSONL files.
"""
import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bam_wrapped_qwen import BAMCLS, BAMQA
from trainer.bam_trainer import BAMTrainer


def parse_parameters():
    parser = argparse.ArgumentParser(description='BAM adapter training')

    # Data
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--label_map_path', type=str)

    # Model
    parser.add_argument('--tokenizer_name', type=str)
    parser.add_argument('--processor_name', type=str)
    parser.add_argument('--training_strategy', type=str, choices=['head_only', 'lora', 'full'])
    parser.add_argument('--device_map', type=str)
    parser.add_argument('--torch_dtype', type=str, choices=['float16', 'float32', 'bfloat16'])

    # LoRA
    parser.add_argument('--lora_r', type=int)
    parser.add_argument('--lora_alpha', type=int)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_target_modules', type=str, nargs='+')

    # Training
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_checkpoint_dir', type=str)
    parser.add_argument('--load_checkpoint_path', type=str)
    parser.add_argument('--validation_result_dir', type=str)
    parser.add_argument('--save_every_n_epochs', type=str)
    parser.add_argument('--save_every_n_steps', type=str)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--num_workers', type=int)

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--no_scheduler', action='store_true')
    parser.add_argument('--scheduler_type', type=str,
                        choices=['linear', 'cosine', 'cosine_with_restarts',
                                 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--warmup_steps', type=int)

    # Validation
    parser.add_argument('--validate_every_n_epochs', type=str)
    parser.add_argument('--validate_every_n_steps', type=str)
    parser.add_argument('--early_stopping_patience', type=int)

    # Dataset
    parser.add_argument('--max_prompt_length', type=int)
    parser.add_argument('--format_prompt', type=str)
    parser.add_argument('--label_key', type=str)
    parser.add_argument('--filter_overlong_prompts', action='store_true')
    parser.add_argument('--truncation', type=str, choices=['left', 'right'])

    # BAM-specific
    parser.add_argument('--use_bam_video', action='store_true')
    parser.add_argument('--use_bam_audio', action='store_true')
    parser.add_argument('--bam_stage', type=str,
                        choices=['bam_only', 'bam_and_classifier_heads_only', 'bam_and_full_model'])
    parser.add_argument('--bam_fresh_start', action='store_true')
    parser.add_argument('--bam_hidden_video', type=int)
    parser.add_argument('--bam_hidden_audio', type=int)
    parser.add_argument('--bam_p_moddrop_video', type=float)
    parser.add_argument('--bam_p_moddrop_audio', type=float)
    parser.add_argument('--d_video_feat', type=int)
    parser.add_argument('--d_audio_feat', type=int)
    parser.add_argument('--bam_video_temporal', type=str,
                        choices=['mean', 'meanstd', 'meanstdp25p75'])
    parser.add_argument('--bam_video_norm', type=str, choices=['none', 'l2', 'zscore'])
    parser.add_argument('--bam_audio_temporal', type=str,
                        choices=['none', 'mean', 'meanstd', 'meanstdp25p75'])
    parser.add_argument('--bam_audio_norm', type=str, choices=['none', 'l2', 'zscore'])
    parser.add_argument('--bam_video_use_ln', action='store_true')
    parser.add_argument('--bam_video_alpha_init', type=float)
    parser.add_argument('--bam_audio_use_ln', action='store_true')
    parser.add_argument('--bam_audio_alpha_init', type=float)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--bam_lr', type=float)
    parser.add_argument('--qa_loss_weight', type=float)

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project', type=str)
    parser.add_argument('--entity', type=str)

    # Task type + dataset identity
    parser.add_argument('--task_type', type=str, choices=['cls', 'qa'], default='cls',
                        help='Model type: cls → BAMCLS, qa → BAMQA')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset identifier (used for logging/checkpointing)')

    # Mode
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--config', type=str, default='configs/config_bam_accelerate.yaml')

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # --- Apply CLI overrides ---
    if args.train_file is not None:         cfg.data.train_file = args.train_file
    if args.val_file is not None:           cfg.data.val_file = args.val_file
    if args.test_file is not None:          cfg.data.test_file = args.test_file
    if args.label_map_path is not None:     cfg.data.label_map_path = args.label_map_path
    if args.tokenizer_name is not None:     cfg.model.tokenizer_name = args.tokenizer_name
    if args.processor_name is not None:     cfg.model.processor_name = args.processor_name
    if args.training_strategy is not None:  cfg.model.training_strategy = args.training_strategy
    if args.device_map is not None:         cfg.model.device_map = args.device_map
    if args.torch_dtype is not None:        cfg.model.torch_dtype = args.torch_dtype
    if args.lora_r is not None:             cfg.model.lora_config.r = args.lora_r
    if args.lora_alpha is not None:         cfg.model.lora_config.alpha = args.lora_alpha
    if args.lora_dropout is not None:       cfg.model.lora_config.dropout = args.lora_dropout
    if args.lora_target_modules is not None: cfg.model.lora_config.target_modules = args.lora_target_modules
    if args.train_batch_size is not None:   cfg.train.train_batch_size = args.train_batch_size
    if args.val_batch_size is not None:     cfg.train.val_batch_size = args.val_batch_size
    if args.test_batch_size is not None:    cfg.train.test_batch_size = args.test_batch_size
    if args.lr is not None:                 cfg.train.lr = args.lr
    if args.epochs is not None:             cfg.train.epochs = args.epochs
    if args.save_checkpoint_dir is not None: cfg.train.save_checkpoint_dir = args.save_checkpoint_dir
    if args.load_checkpoint_path is not None: cfg.train.load_checkpoint_path = args.load_checkpoint_path
    if args.validation_result_dir is not None: cfg.train.validation_result_dir = args.validation_result_dir
    if args.save_every_n_epochs is not None: cfg.train.save_every_n_epochs = args.save_every_n_epochs
    if args.save_every_n_steps is not None:  cfg.train.save_every_n_steps = args.save_every_n_steps
    if args.gradient_accumulation_steps is not None: cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_workers is not None:        cfg.train.num_workers = args.num_workers
    if args.use_scheduler:                  cfg.train.use_scheduler = True
    if args.no_scheduler:                   cfg.train.use_scheduler = False
    if args.scheduler_type is not None:     cfg.train.scheduler_type = args.scheduler_type
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = None if args.warmup_steps == -1 else args.warmup_steps
    if args.base_lr is not None:            cfg.train.base_lr = args.base_lr
    if args.bam_lr is not None:             cfg.train.bam_lr = args.bam_lr
    if args.qa_loss_weight is not None:     cfg.train.qa_loss_weight = args.qa_loss_weight
    if args.validate_every_n_epochs is not None: cfg.train.validate_every_n_epochs = args.validate_every_n_epochs
    if args.validate_every_n_steps is not None:  cfg.train.validate_every_n_steps = args.validate_every_n_steps
    if args.early_stopping_patience is not None: cfg.train.early_stopping_patience = args.early_stopping_patience
    if args.max_prompt_length is not None:  cfg.dataset_config.max_prompt_length = args.max_prompt_length
    if args.format_prompt is not None:      cfg.dataset_config.format_prompt = args.format_prompt
    if args.label_key is not None:          cfg.dataset_config.label_key = args.label_key
    if args.filter_overlong_prompts:        cfg.dataset_config.filter_overlong_prompts = True
    if args.truncation is not None:         cfg.dataset_config.truncation = args.truncation
    if args.use_wandb:                      cfg.wandb.use = True
    if args.project is not None:            cfg.wandb.project = args.project
    if args.entity is not None:             cfg.wandb.entity = args.entity
    if args.mode is not None:               cfg.mode = args.mode

    if not hasattr(cfg, 'bam'):
        cfg.bam = OmegaConf.create({})
    cfg.bam.task_type    = args.task_type
    cfg.bam.dataset_name = args.dataset_name or ""
    if args.use_bam_video:                          cfg.bam.use_bam_video = True
    if args.use_bam_audio:                          cfg.bam.use_bam_audio = True
    if args.bam_stage is not None:                  cfg.bam.bam_stage = args.bam_stage
    if args.bam_fresh_start:                        cfg.bam.fresh_start = True
    if args.bam_hidden_video is not None:           cfg.bam.bam_hidden_video = args.bam_hidden_video
    if args.bam_hidden_audio is not None:           cfg.bam.bam_hidden_audio = args.bam_hidden_audio
    if args.bam_p_moddrop_video is not None:        cfg.bam.bam_p_moddrop_video = args.bam_p_moddrop_video
    if args.bam_p_moddrop_audio is not None:        cfg.bam.bam_p_moddrop_audio = args.bam_p_moddrop_audio
    if args.d_video_feat is not None:               cfg.bam.d_video_feat = args.d_video_feat
    if args.d_audio_feat is not None:               cfg.bam.d_audio_feat = args.d_audio_feat
    if args.bam_video_temporal is not None:         cfg.bam.video_temporal = args.bam_video_temporal
    if args.bam_video_norm is not None:             cfg.bam.video_norm = args.bam_video_norm
    if args.bam_audio_temporal is not None:         cfg.bam.audio_temporal = args.bam_audio_temporal
    if args.bam_audio_norm is not None:             cfg.bam.audio_norm = args.bam_audio_norm
    if args.bam_video_use_ln:                       cfg.bam.video_use_ln = True
    if args.bam_video_alpha_init is not None:       cfg.bam.video_alpha_init = args.bam_video_alpha_init
    if args.bam_audio_use_ln:                       cfg.bam.audio_use_ln = True
    if args.bam_audio_alpha_init is not None:       cfg.bam.audio_alpha_init = args.bam_audio_alpha_init

    def _parse_n(v):
        if v is None or v == "None":
            return None
        return int(v)

    with open(cfg.data.label_map_path, 'r') as f:
        label_config = json.load(f)

    torch_dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    bam = cfg.bam

    params = {
        'train_data_file':              cfg.data.train_file,
        'val_data_file':                cfg.data.val_file,
        'test_data_file':               cfg.data.test_file,
        'label_map_path':               cfg.data.label_map_path,
        'tokenizer_name':               cfg.model.tokenizer_name,
        'processor_name':               cfg.model.processor_name,
        'training_strategy':            cfg.model.training_strategy,
        'device_map':                   cfg.model.device_map,
        'torch_dtype':                  torch_dtype_map.get(cfg.model.torch_dtype, torch.float16),
        'train_batch_size':             int(cfg.train.train_batch_size),
        'val_batch_size':               int(cfg.train.val_batch_size),
        'test_batch_size':              int(cfg.train.test_batch_size),
        'lr':                           float(cfg.train.lr),
        'epochs':                       int(cfg.train.epochs),
        'save_checkpoint_dir':          cfg.train.save_checkpoint_dir,
        'load_checkpoint_path':         cfg.train.load_checkpoint_path,
        'validation_result_dir':        cfg.train.validation_result_dir,
        'save_every_n_epochs':          _parse_n(cfg.train.save_every_n_epochs),
        'save_every_n_steps':           _parse_n(cfg.train.save_every_n_steps),
        'gradient_accumulation_steps':  int(cfg.train.gradient_accumulation_steps),
        'num_workers':                  int(cfg.train.num_workers),
        'use_scheduler':                bool(cfg.train.use_scheduler),
        'scheduler_type':               cfg.train.scheduler_type,
        'warmup_steps':                 cfg.train.warmup_steps,
        'validate_every_n_epochs':      _parse_n(cfg.train.validate_every_n_epochs),
        'validate_every_n_steps':       _parse_n(cfg.train.validate_every_n_steps),
        'early_stopping_patience':      int(cfg.train.early_stopping_patience),
        'use_wandb':                    bool(cfg.wandb.use),
        'wandb_project':                cfg.wandb.project,
        'wandb_entity':                 cfg.wandb.entity,
        'full_label_scheme':            label_config,
        'label_map':                    label_config["label_mapping"],
        'num_classes':                  label_config["num_classes"],
        'lora_config': {
            'r':               int(cfg.model.lora_config.r),
            'alpha':           int(cfg.model.lora_config.alpha),
            'dropout':         float(cfg.model.lora_config.dropout),
            'target_modules':  list(cfg.model.lora_config.target_modules),
        },
        'dataset_config':               OmegaConf.create(dict(cfg.dataset_config)),
        'mode':                         getattr(cfg, 'mode', args.mode),
    }
    return params, label_config, bam, cfg


def main():
    params, label_config, bam, cfg = parse_parameters()

    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer_name'])
    processor = AutoProcessor.from_pretrained(params['processor_name'])

    task_type    = bam.get("task_type", "cls")
    dataset_name = bam.get("dataset_name", "")

    model_cls = BAMCLS if task_type == "cls" else BAMQA
    print(f"[INFO] Initializing {model_cls.__name__} (task_type={task_type}, dataset={dataset_name}) "
          f"with {params['num_classes']} classes")
    model = model_cls(
        full_label_scheme=params['full_label_scheme'],
        freeze_backbone=params['training_strategy'],
        lora_config=params['lora_config'] if params['training_strategy'] == "lora" else None,
        device_map=params['device_map'],
        torch_dtype=params['torch_dtype'],
    )

    def _bam(key, default=None):
        return getattr(bam, key, default)

    global_config = {
        'TOKENIZER_NAME':               params['tokenizer_name'],
        'TRAINING_STRATEGY':            params['training_strategy'],
        'FULL_LABEL_SCHEME':            params['full_label_scheme'],
        'LABEL_MAP':                    params['label_map'],
        'LABEL_MAP_PATH':               params['label_map_path'],
        'NUM_CLASSES':                  params['num_classes'],
        'LORA_CONFIG':                  params['lora_config'],
        'label_config':                 label_config,
        'VALIDATION_RESULT_DIR':        params['validation_result_dir'],
        'USE_SCHEDULER':                params['use_scheduler'],
        'SCHEDULER_TYPE':               params['scheduler_type'],
        'WARMUP_STEPS':                 params['warmup_steps'],
        'VALIDATE_EVERY_N_EPOCHS':      params['validate_every_n_epochs'],
        'VALIDATE_EVERY_N_STEPS':       params['validate_every_n_steps'],
        'SAVE_EVERY_N_EPOCHS':          params['save_every_n_epochs'],
        'SAVE_EVERY_N_STEPS':           params['save_every_n_steps'],
        'EARLY_STOPPING_PATIENCE':      params['early_stopping_patience'],
        'USE_WANDB':                    params['use_wandb'],
        'WANDB_PROJECT':                params['wandb_project'],
        'WANDB_ENTITY':                 params['wandb_entity'],
        # Task identity
        'TASK_TYPE':                    task_type,
        'DATASET_NAME':                 dataset_name,
        # BAM-specific
        'USE_BAM_VIDEO':                bool(_bam('use_bam_video', False)),
        'USE_BAM_AUDIO':                bool(_bam('use_bam_audio', False)),
        'BAM_STAGE':                    _bam('bam_stage', 'bam_only'),
        'BAM_FRESH_START':              bool(_bam('fresh_start', False)),
        'BAM_HIDDEN':                   int(_bam('bam_hidden', 128)),
        'BAM_HIDDEN_VIDEO':             int(_bam('bam_hidden_video', 128)),
        'BAM_HIDDEN_AUDIO':             int(_bam('bam_hidden_audio', 128)),
        'BAM_P_MODDROP_VIDEO':          float(_bam('bam_p_moddrop_video', 0.30)),
        'BAM_P_MODDROP_AUDIO':          float(_bam('bam_p_moddrop_audio', 0.30)),
        'D_VIDEO_FEAT':                 _bam('d_video_feat', None),
        'D_AUDIO_FEAT':                 _bam('d_audio_feat', None),
        'BAM_VIDEO_TEMPORAL':           _bam('video_temporal', 'meanstd'),
        'BAM_VIDEO_NORM':               _bam('video_norm', None),
        'BAM_AUDIO_TEMPORAL':           _bam('audio_temporal', 'none'),
        'BAM_AUDIO_NORM':               _bam('audio_norm', 'l2'),
        'BAM_VIDEO_USE_LN':             bool(_bam('video_use_ln', False)),
        'BAM_VIDEO_ALPHA_INIT':         float(_bam('video_alpha_init', 1.0)),
        'BAM_AUDIO_USE_LN':             bool(_bam('audio_use_ln', False)),
        'BAM_AUDIO_ALPHA_INIT':         float(_bam('audio_alpha_init', 1.0)),
        'BASE_LR':                      float(getattr(cfg.train, 'base_lr', params['lr'] * 0.25)),
        'BAM_LR':                       float(getattr(cfg.train, 'bam_lr', params['lr'] * 5.0)),
        'QA_LOSS_WEIGHT':               float(getattr(cfg.train, 'qa_loss_weight', 1.0)),
    }

    trainer = BAMTrainer(
        data_files=params['train_data_file'],
        val_data_files=params['val_data_file'],
        test_data_files=params['test_data_file'],
        tokenizer=tokenizer,
        processor=processor,
        config=params['dataset_config'],
        batch_size=params['train_batch_size'],
        val_batch_size=params['val_batch_size'],
        test_batch_size=params['test_batch_size'],
        lr=params['lr'],
        epochs=params['epochs'],
        save_checkpoint_dir=params['save_checkpoint_dir'],
        load_checkpoint_path=params['load_checkpoint_path'],
        model=model,
        gradient_accumulation_steps=params['gradient_accumulation_steps'],
        num_workers=params['num_workers'],
        global_config=global_config,
    )

    mode = params['mode']
    if mode == 'train':
        trainer.train()
    elif mode == 'test':
        trainer.test()
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
