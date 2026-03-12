import time
from .wandb_utils import log_metrics


def log_batch_training_metrics(epoch, batch_idx, total_batches, loss,
                        correct, total, epoch_start_time, start_time, 
                        gradient_accumulation_steps, batch_size, epochs, 
                        accelerator, use_wandb, current_lr=None, current_step=None):
    """Log training metrics to wandb."""
    if not use_wandb or not accelerator.is_main_process:
        return
    
    # Log batch metrics at effective batch size steps
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        avg_effective_loss = loss / max(1, total)
        batch_info = {
            'effective_batch_loss': loss,
            'avg_effective_batch_loss': avg_effective_loss,
            'effective_batch_correct': correct,
            'effective_batch_total': total,
            'effective_batch_accuracy': correct / total,
            'batch_idx': batch_idx,
            'effective_batch_size': batch_size * gradient_accumulation_steps * accelerator.num_processes,
        }
        
        # Add learning rate if provided
        if current_lr is not None:
            batch_info['learning_rate'] = float(current_lr)
        
        if not current_step:
            current_step = (epoch * total_batches) + batch_idx + 1
        log_metrics('effective_batch_metrics', batch_info, step=current_step)

    # Log training progress statistics
    batch_progress = (batch_idx + 1) / total_batches
    epoch_progress = (epoch + 1) / epochs
    overall_progress = ((epoch * total_batches) + batch_idx + 1) / (epochs * total_batches)
    
    # Calculate time statistics
    elapsed_time = time.time() - epoch_start_time
    total_elapsed = time.time() - start_time if hasattr(start_time, '__call__') else elapsed_time
    
    # Calculate ETA
    if batch_progress > 0:
        epoch_eta = (elapsed_time / batch_progress) * (1 - batch_progress)
        overall_eta = (total_elapsed / overall_progress) * (1 - overall_progress) if overall_progress > 0 else 0
    else:
        epoch_eta = 0
        overall_eta = 0
    
    # Calculate training rate
    training_rate = (batch_idx + 1) / max(1, elapsed_time)
    
    # Log progress statistics
    progress_stats = {
        'batch_progress': batch_progress,
        'epoch_progress': epoch_progress,
        'overall_progress': overall_progress,
        'epoch_elapsed_time_seconds': elapsed_time,
        'epoch_eta_seconds': epoch_eta,
        'overall_eta_seconds': overall_eta,
        'training_rate': training_rate,
        'current_epoch': epoch + 1,
        'total_epochs': epochs,
        'current_batch': batch_idx + 1,
        'total_batches': total_batches
    }
    if not current_step:
        current_step = (epoch * total_batches) + batch_idx + 1
    log_metrics('training_progress', progress_stats, step=current_step)


def log_validation_results(val_results, current_step, split_name, accelerator, use_wandb):
    """Log validation results to wandb."""
    if not use_wandb or not accelerator.is_main_process or val_results is None:
        return
    
    # Log validation metrics
    vm = {
        'loss': val_results['loss'], 
        'best_val_f1': val_results.get('best_val_f1', 0.0), 
        'steps_without_improvement': val_results.get('steps_without_improvement', 0)
    }
    
    # Add aggregate metrics
    for key, value in val_results['aggregate_metrics'].items():
        vm[key] = value
    
    # logging of aggregate metrics
    log_metrics('val', vm, step=current_step)
    
    # logging of per-dataset metrics
    if 'per_dataset_metrics' in val_results['evaluation_results']:
            per_dataset_metrics = val_results['evaluation_results']['per_dataset_metrics']
            
            # Group metrics by dataset for better wandb visualization
            dataset_metrics = {}
            for metric_key, value in per_dataset_metrics.items():
                if '/' in metric_key:
                    dataset_name, metric_name = metric_key.split('/', 1)
                    if dataset_name not in dataset_metrics:
                        dataset_metrics[dataset_name] = {}
                    dataset_metrics[dataset_name][metric_name] = value
            
            # Log each dataset separately
            for dataset_name, metrics in dataset_metrics.items():
                log_metrics(f'dataset_{dataset_name}', metrics, step=current_step)


def log_epoch_training_metrics(epoch, avg_train_loss, train_acc, total_batches, accelerator, use_wandb, current_step=None):
    """Log epoch-level training metrics."""
    if not use_wandb or not accelerator.is_main_process:
        return
    
    if not current_step:    
        current_step = (epoch + 1) * total_batches
        
    log_metrics('train', {
        'loss': avg_train_loss,
        'accuracy': train_acc
    }, step=current_step)
