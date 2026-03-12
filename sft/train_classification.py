import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from models.qwen2_5_omni_classifier_heads_decoder import MultiHeadOmniClassifier
from trainer.cls_trainer import CLSTrainer

def parse_parameters():
    """
    Parse parameters from YAML config file with command-line argument overrides.
    
    Returns:
        dict: Dictionary containing all parsed parameters
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train OmniClassifier with Accelerate')
    
    # Data parameters
    parser.add_argument('--train_file', type=str, help='Training data file path')
    parser.add_argument('--val_file', type=str, help='Validation data file path')
    parser.add_argument('--test_file', type=str, help='Test data file path')
    parser.add_argument('--label_map_path', type=str, help='Path to label mapping JSON file')
    
    # Model parameters
    parser.add_argument('--tokenizer_name', type=str, help='Tokenizer model name')
    parser.add_argument('--processor_name', type=str, help='Processor model name')
    parser.add_argument('--training_strategy', type=str, 
                       choices=['head_only', 'lora', 'full'],
                       help='Training strategy: head_only, lora, or full')
    parser.add_argument('--device_map', type=str, help='Device mapping (auto, cpu, or specific devices)')
    parser.add_argument('--torch_dtype', type=str, 
                       choices=['float16', 'float32', 'bfloat16'],
                       help='PyTorch data type')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                       help='LoRA target modules (space-separated list)')
    
    # Training parameters
    parser.add_argument('--train_batch_size', type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, help='Validation batch size')
    parser.add_argument('--test_batch_size', type=int, help='Test batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--save_checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint_path', type=str, help='Path to load checkpoint from')
    parser.add_argument('--validation_result_dir', type=str, help='Directory to save validation results')
    parser.add_argument('--save_every_n_epochs', type=str, help='Save checkpoint every N epochs')
    parser.add_argument('--save_every_n_steps', type=str, help='Save checkpoint every N steps (use "None" to disable)')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers')
    
    # Scheduler parameters
    parser.add_argument('--use_scheduler', action='store_true', help='Enable learning rate scheduler')
    parser.add_argument('--no_scheduler', action='store_true', help='Disable learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, 
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
                       help='Type of learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, help='Number of warmup steps (set to -1 to disable)')
    
    # Validation parameters
    parser.add_argument('--validate_every_n_epochs', type=str, 
                       help='Validate every N epochs (use "None" to disable)')
    parser.add_argument('--validate_every_n_steps', type=str, 
                       help='Validate every N steps (use "None" to disable)')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience')
    
    # Dataset parameters
    parser.add_argument('--max_prompt_length', type=int, help='Maximum prompt length')
    parser.add_argument('--modalities', type=str, help='Comma-separated list of modalities')
    parser.add_argument('--prompt_key', type=str, help='Prompt key in dataset')
    parser.add_argument('--image_key', type=str, help='Image key in dataset')
    parser.add_argument('--video_key', type=str, help='Video key in dataset')
    parser.add_argument('--audio_key', type=str, help='Audio key in dataset')
    parser.add_argument('--label_key', type=str, help='Label key in dataset')
    parser.add_argument('--return_multi_modal_inputs', action='store_true', help='Return multi-modal inputs')
    parser.add_argument('--filter_overlong_prompts', action='store_true', help='Filter overlong prompts')
    parser.add_argument('--truncation', type=str, choices=['left', 'right'], help='Truncation direction')
    parser.add_argument('--format_prompt', type=str, help='Path to format prompt template')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project', type=str, help='Wandb project name')
    parser.add_argument('--entity', type=str, help='Wandb entity name')
    
    # System parameters
    parser.add_argument('--cuda_visible_devices', type=str, help='CUDA visible devices')
    
    # Mode parameter
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/config_accelerate.yaml', 
                       help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load base configuration from YAML file
    cfg = OmegaConf.load(args.config)
    
    # Override with command line arguments
    # Data parameters
    if args.train_file is not None:
        cfg.data.train_file = args.train_file
    if args.val_file is not None:
        cfg.data.val_file = args.val_file
    if args.test_file is not None:
        cfg.data.test_file = args.test_file
    if args.label_map_path is not None:
        cfg.data.label_map_path = args.label_map_path
    
    # Model parameters
    if args.tokenizer_name is not None:
        cfg.model.tokenizer_name = args.tokenizer_name
    if args.processor_name is not None:
        cfg.model.processor_name = args.processor_name
    if args.training_strategy is not None:
        cfg.model.training_strategy = args.training_strategy
    if args.device_map is not None:
        cfg.model.device_map = args.device_map
    if args.torch_dtype is not None:
        cfg.model.torch_dtype = args.torch_dtype
    
    # LoRA parameters
    if args.lora_r is not None:
        cfg.model.lora_config.r = args.lora_r
    if args.lora_alpha is not None:
        cfg.model.lora_config.alpha = args.lora_alpha
    if args.lora_dropout is not None:
        cfg.model.lora_config.dropout = args.lora_dropout
    if args.lora_target_modules is not None:
        cfg.model.lora_config.target_modules = args.lora_target_modules
    
    # Training parameters
    if args.train_batch_size is not None:
        cfg.train.train_batch_size = args.train_batch_size
    if args.val_batch_size is not None:
        cfg.train.val_batch_size = args.val_batch_size
    if args.test_batch_size is not None:
        cfg.train.test_batch_size = args.test_batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.save_checkpoint_dir is not None:
        cfg.train.save_checkpoint_dir = args.save_checkpoint_dir
    if args.load_checkpoint_path is not None:
        cfg.train.load_checkpoint_path = args.load_checkpoint_path
    if args.validation_result_dir is not None:
        cfg.train.validation_result_dir = args.validation_result_dir
    if args.save_every_n_epochs is not None:
        cfg.train.save_every_n_epochs = args.save_every_n_epochs
    if args.save_every_n_steps is not None:
        cfg.train.save_every_n_steps = args.save_every_n_steps
    if args.gradient_accumulation_steps is not None:
        cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    
    # Scheduler parameters
    if args.use_scheduler:
        cfg.train.use_scheduler = True
    if args.no_scheduler:
        cfg.train.use_scheduler = False
    if args.scheduler_type is not None:
        cfg.train.scheduler_type = args.scheduler_type
    if args.warmup_steps is not None:
        if args.warmup_steps == -1:
            cfg.train.warmup_steps = None
        else:
            cfg.train.warmup_steps = args.warmup_steps
    
    # Validation parameters
    if args.validate_every_n_epochs is not None:
        cfg.train.validate_every_n_epochs = args.validate_every_n_epochs
    if args.validate_every_n_steps is not None:
        cfg.train.validate_every_n_steps = args.validate_every_n_steps
    if args.early_stopping_patience is not None:
        cfg.train.early_stopping_patience = args.early_stopping_patience
    
    # Dataset parameters
    if args.max_prompt_length is not None:
        cfg.dataset_config.max_prompt_length = args.max_prompt_length
    if args.modalities is not None:
        cfg.dataset_config.modalities = args.modalities
    if args.prompt_key is not None:
        cfg.dataset_config.prompt_key = args.prompt_key
    if args.image_key is not None:
        cfg.dataset_config.image_key = args.image_key
    if args.video_key is not None:
        cfg.dataset_config.video_key = args.video_key
    if args.audio_key is not None:
        cfg.dataset_config.audio_key = args.audio_key
    if args.label_key is not None:
        cfg.dataset_config.label_key = args.label_key
    if args.return_multi_modal_inputs:
        cfg.dataset_config.return_multi_modal_inputs = True
    if args.filter_overlong_prompts:
        cfg.dataset_config.filter_overlong_prompts = True
    if args.truncation is not None:
        cfg.dataset_config.truncation = args.truncation
    if args.format_prompt is not None:
        cfg.dataset_config.format_prompt = args.format_prompt
    cfg.dataset_config.task_filter = "cls"

    # Wandb parameters
    if args.use_wandb:
        cfg.wandb.use = True
    if args.project is not None:
        cfg.wandb.project = args.project
    if args.entity is not None:
        cfg.wandb.entity = args.entity
    
    # System parameters
    if args.cuda_visible_devices is not None:
        if not hasattr(cfg, 'system'):
            cfg.system = OmegaConf.create({})
        cfg.system.cuda_visible_devices = args.cuda_visible_devices
    
    # Mode parameter
    if args.mode is not None:
        if not hasattr(cfg, 'mode'):
            cfg.mode = args.mode
        else:
            cfg.mode = args.mode
    
    # Parse all parameters
    params = {}
    
    # Data parameters
    params['train_data_file'] = cfg.data.train_file
    params['val_data_file'] = cfg.data.val_file
    params['test_data_file'] = cfg.data.test_file
    params['label_map_path'] = cfg.data.label_map_path
    
    # Model parameters
    params['tokenizer_name'] = cfg.model.tokenizer_name
    params['processor_name'] = cfg.model.processor_name
    params['training_strategy'] = cfg.model.training_strategy
    params['device_map'] = cfg.model.device_map
    params['torch_dtype_str'] = cfg.model.torch_dtype
    
    # Convert torch_dtype string to actual torch dtype
    if params['torch_dtype_str'] == "float16":
        params['torch_dtype'] = torch.float16
    elif params['torch_dtype_str'] == "float32":
        params['torch_dtype'] = torch.float32
    elif params['torch_dtype_str'] == "bfloat16":
        params['torch_dtype'] = torch.bfloat16
    else:
        params['torch_dtype'] = torch.float16  # default
    
    # Training parameters
    params['train_batch_size'] = cfg.train.train_batch_size
    params['val_batch_size'] = cfg.train.val_batch_size
    params['test_batch_size'] = cfg.train.test_batch_size
    params['lr'] = float(cfg.train.lr)
    params['epochs'] = int(cfg.train.epochs)
    params['save_checkpoint_dir'] = cfg.train.save_checkpoint_dir
    params['load_checkpoint_path'] = cfg.train.load_checkpoint_path
    params['validation_result_dir'] = cfg.train.validation_result_dir
    params['save_every_n_epochs'] = cfg.train.save_every_n_epochs
    params['save_every_n_steps'] = cfg.train.save_every_n_steps

    if params['save_every_n_steps'] is not None:
        if params['save_every_n_steps'] == "None":
            params['save_every_n_steps'] = None
        else:
            params['save_every_n_steps'] = int(params['save_every_n_steps'])

    if params['save_every_n_epochs'] is not None:
        if params['save_every_n_epochs'] == "None":
            params['save_every_n_epochs'] = None
        else:
            params['save_every_n_epochs'] = int(params['save_every_n_epochs'])

    params['gradient_accumulation_steps'] = int(cfg.train.gradient_accumulation_steps)
    params['num_workers'] = int(cfg.train.num_workers)
    
    # Scheduler configuration
    params['use_scheduler'] = bool(cfg.train.use_scheduler)
    params['scheduler_type'] = cfg.train.scheduler_type
    params['warmup_steps'] = cfg.train.warmup_steps
    
    # Validation configuration
    params['validate_every_n_epochs'] = cfg.train.validate_every_n_epochs
    params['validate_every_n_steps'] = cfg.train.validate_every_n_steps
    if params['validate_every_n_steps'] is not None:
        if params['validate_every_n_steps'] == "None":
            params['validate_every_n_steps'] = None
        else:
            params['validate_every_n_steps'] = int(params['validate_every_n_steps'])
    if params['validate_every_n_epochs'] is not None:
        if params['validate_every_n_epochs'] == "None":
            params['validate_every_n_epochs'] = None
        else:
            params['validate_every_n_epochs'] = int(params['validate_every_n_epochs'])

    params['save_best_model'] = True
    params['early_stopping_patience'] = int(cfg.train.early_stopping_patience)
    
    # Wandb configuration
    params['use_wandb'] = bool(cfg.wandb.use)
    params['wandb_project'] = cfg.wandb.project
    params['wandb_entity'] = cfg.wandb.entity
    
    # Load label mapping from JSON file
    with open(params['label_map_path'], 'r') as f:
        label_config = json.load(f)
    
    params['full_label_scheme'] = label_config
    params['label_map'] = label_config["label_mapping"]
    params['num_classes'] = label_config["num_classes"]
    

    # LoRA Configuration (only used when training_strategy = "lora")
    params['lora_config'] = {
        'r': int(cfg.model.lora_config.r),
        'alpha': int(cfg.model.lora_config.alpha),
        'dropout': float(cfg.model.lora_config.dropout),
        'target_modules': list(cfg.model.lora_config.target_modules),
    }
    
    # Dataset config
    params['dataset_config'] = OmegaConf.create(dict(cfg.dataset_config))
    
    # Mode parameter
    params['mode'] = getattr(cfg, 'mode', 'train')
    
    # Print configuration summary
    print(f"[INFO] Mode: {params['mode']}")
    print(f"[INFO] Training strategy: {params['training_strategy']}")
    print(f"[INFO] Loaded label mapping with {params['num_classes']} classes from {params['label_map_path']}")
    print(f"[INFO] Available datasets: {', '.join(label_config['datasets'])}")
    print(f"[INFO] Gradient accumulation: {params['gradient_accumulation_steps']} steps (effective batch size: {params['train_batch_size'] * params['gradient_accumulation_steps']})")
    print(f"[INFO] Data loading: {params['num_workers']} worker processes (0 = single-threaded, {params['num_workers']}+ = multi-threaded)")
    print(f"[INFO] Learning rate: {params['lr']}")
    print(f"[INFO] Epochs: {params['epochs']}")
    if params['mode'] == 'train':
        print(f"[INFO] Save checkpoint dir: {params['save_checkpoint_dir']}")
        print(f"[INFO] Save every N epochs: {params['save_every_n_epochs']}")
        print(f"[INFO] Save every N steps: {params['save_every_n_steps']}")
        print(f"[INFO] Scheduler: {'Enabled' if params['use_scheduler'] else 'Disabled'}")
        if params['use_scheduler']:
            print(f"[INFO] Scheduler type: {params['scheduler_type']}")
            print(f"[INFO] Warmup steps: {params['warmup_steps']}")
        print(f"[INFO] Validate every N epochs: {params['validate_every_n_epochs']}")
        print(f"[INFO] Validate every N steps: {params['validate_every_n_steps']}")
        print(f"[INFO] Early stopping patience: {params['early_stopping_patience']}")
    if params['load_checkpoint_path']:
        print(f"[INFO] Load checkpoint path: {params['load_checkpoint_path']}")
    print(f"[INFO] Wandb project: {params['wandb_project']}")
    
    return params, label_config

def main():
    # Parse parameters
    params, label_config = parse_parameters()
    
    # Extract parameters to global variables for backward compatibility
    TRAIN_DATA_FILE = params['train_data_file']
    VAL_DATA_FILE = params['val_data_file']
    TEST_DATA_FILE = params['test_data_file']
    TOKENIZER_NAME = params['tokenizer_name']
    PROCESSOR_NAME = params['processor_name']
    TRAINING_STRATEGY = params['training_strategy']
    DEVICE_MAP = params['device_map']
    TORCH_DTYPE = params['torch_dtype']
    TRAIN_BATCH_SIZE = params['train_batch_size']
    VAL_BATCH_SIZE = params['val_batch_size']
    TEST_BATCH_SIZE = params['test_batch_size']
    LR = params['lr']
    EPOCHS = params['epochs']
    SAVE_CHECKPOINT_DIR = params['save_checkpoint_dir']
    LOAD_CHECKPOINT_PATH = params['load_checkpoint_path']
    VALIDATION_RESULT_DIR = params['validation_result_dir']
    SAVE_EVERY_N_EPOCHS = params['save_every_n_epochs']
    SAVE_EVERY_N_STEPS = params['save_every_n_steps']
    GRADIENT_ACCUMULATION_STEPS = params['gradient_accumulation_steps']
    NUM_WORKERS = params['num_workers']
    VALIDATE_EVERY_N_EPOCHS = params['validate_every_n_epochs']
    VALIDATE_EVERY_N_STEPS = params['validate_every_n_steps']
    SAVE_BEST_MODEL = params['save_best_model']
    EARLY_STOPPING_PATIENCE = params['early_stopping_patience']
    USE_WANDB = params['use_wandb']
    WANDB_PROJECT = params['wandb_project']
    WANDB_ENTITY = params['wandb_entity']
    FULL_LABEL_SCHEME = params['full_label_scheme']
    LABEL_MAP = params['label_map']
    LABEL_MAP_PATH = params['label_map_path']
    NUM_CLASSES = params['num_classes']
    LORA_CONFIG = params['lora_config']
    config = params['dataset_config']
    MODE = params['mode']


    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
    
    print(f"[INFO] Initializing model with {NUM_CLASSES} classes")
    
    # Initialize model with training strategy
    print(f"[INFO] Initializing OmniClassifier with training strategy: {TRAINING_STRATEGY}")
    model = MultiHeadOmniClassifier(
        full_label_scheme=FULL_LABEL_SCHEME,
        freeze_backbone=TRAINING_STRATEGY,
        lora_config=LORA_CONFIG if TRAINING_STRATEGY == "lora" else None,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE
    )
    # Print trainable parameters info
    model.get_trainable_parameters()
    
    # Create global config dictionary for trainer
    global_config = {
        'TRAIN_DATA_FILE': TRAIN_DATA_FILE,
        'VAL_DATA_FILE': VAL_DATA_FILE,
        'TEST_DATA_FILE': TEST_DATA_FILE,
        'TOKENIZER_NAME': TOKENIZER_NAME,
        'PROCESSOR_NAME': PROCESSOR_NAME,
        'TRAINING_STRATEGY': TRAINING_STRATEGY,
        'DEVICE_MAP': DEVICE_MAP,
        'TORCH_DTYPE': TORCH_DTYPE,
        'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
        'VAL_BATCH_SIZE': VAL_BATCH_SIZE,
        'TEST_BATCH_SIZE': TEST_BATCH_SIZE,
        'LR': LR,
        'EPOCHS': EPOCHS,
        'SAVE_CHECKPOINT_DIR': SAVE_CHECKPOINT_DIR,
        'LOAD_CHECKPOINT_PATH': LOAD_CHECKPOINT_PATH,
        'VALIDATION_RESULT_DIR': VALIDATION_RESULT_DIR,
        'SAVE_EVERY_N_EPOCHS': SAVE_EVERY_N_EPOCHS,
        'SAVE_EVERY_N_STEPS': SAVE_EVERY_N_STEPS,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'NUM_WORKERS': NUM_WORKERS,
        'VALIDATE_EVERY_N_EPOCHS': VALIDATE_EVERY_N_EPOCHS,
        'VALIDATE_EVERY_N_STEPS': VALIDATE_EVERY_N_STEPS,
        'SAVE_BEST_MODEL': SAVE_BEST_MODEL,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'USE_WANDB': USE_WANDB,
        'WANDB_PROJECT': WANDB_PROJECT,
        'WANDB_ENTITY': WANDB_ENTITY,
        'LABEL_MAP': LABEL_MAP,
        'LABEL_MAP_PATH': LABEL_MAP_PATH,
        'NUM_CLASSES': NUM_CLASSES,
        'LORA_CONFIG': LORA_CONFIG,
        'label_config': label_config,
        'FULL_LABEL_SCHEME': FULL_LABEL_SCHEME,
        'USE_SCHEDULER': params['use_scheduler'],
        'SCHEDULER_TYPE': params['scheduler_type'],
        'WARMUP_STEPS': params['warmup_steps'],
        'MODE': MODE
    }

    trainer = CLSTrainer(
        data_files=TRAIN_DATA_FILE,
        val_data_files=VAL_DATA_FILE,
        test_data_files=TEST_DATA_FILE,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        save_checkpoint_dir=SAVE_CHECKPOINT_DIR,
        load_checkpoint_path=LOAD_CHECKPOINT_PATH,
        model=model,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_workers=NUM_WORKERS,
        global_config=global_config
    )
    
    # Execute based on mode
    if MODE == 'train':
        print(f"[INFO] Starting training mode...")
        trainer.train()
    elif MODE == 'test':
        print(f"[INFO] Starting testing mode...")
        test_results = trainer.test()
        print(f"[INFO] Test results: {test_results}")
    else:
        raise ValueError(f"Invalid mode: {MODE}. Must be 'train' or 'test'.")

if __name__ == "__main__":
    main()
