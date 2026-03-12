import wandb
from tqdm import tqdm
import time


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds < 0:
        return "00:00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_percentage(value, total):
    """Convert value/total to percentage string."""
    if total == 0:
        return "0.00%"
    percentage = (value / total) * 100
    return f"{percentage:.2f}%"


class WandbTqdmCallback:
    """Custom tqdm callback to log progress to wandb."""
    
    def __init__(self, name: str, total: int, log_interval: int = 1):
        self.name = name
        self.total = total
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def __call__(self, n: int = 1):
        self.current += n
        current_time = time.time()
        
        # Log to wandb at specified intervals
        if (self.current % self.log_interval == 0 or 
            self.current == self.total or 
            current_time - self.last_log_time >= 5.0):  # Log at least every 5 seconds
            
            if wandb.run:
                progress = self.current / self.total if self.total > 0 else 0
                elapsed_time = current_time - self.start_time
                eta = (elapsed_time / max(1, self.current)) * (self.total - self.current) if self.current > 0 else 0
                rate = self.current / max(1, elapsed_time)
                
                # Format values for better readability
                progress_percentage = format_percentage(self.current, self.total)
                elapsed_time_formatted = format_time(elapsed_time)
                eta_formatted = format_time(eta)
                
                wandb.log({
                    f"progress/{self.name}": {
                        "current": self.current,
                        "total": self.total,
                        "progress": progress,
                        "progress_percentage": progress_percentage,
                        "elapsed_time": elapsed_time,
                        "elapsed_time_formatted": elapsed_time_formatted,
                        "eta": eta,
                        "eta_formatted": eta_formatted,
                        "rate": rate,
                        "rate_per_second": f"{rate:.2f}/s",
                        "summary": f"{progress_percentage} complete | Elapsed: {elapsed_time_formatted} | ETA: {eta_formatted} | Rate: {rate:.2f}/s"
                    }
                })
            
            self.last_log_time = current_time


def create_wandb_tqdm(iterable=None, desc=None, total=None, **kwargs):
    """Create a tqdm progress bar that logs to wandb."""
    if not wandb.run:
        # If wandb is not initialized, return regular tqdm
        return tqdm(iterable=iterable, desc=desc, total=total, **kwargs)
    
    # Create custom callback for wandb logging
    callback = WandbTqdmCallback(name=desc or "progress", total=total)
    
    # Create tqdm with custom callback
    pbar = tqdm(iterable=iterable, desc=desc, total=total, **kwargs)
    
    # Store callback for later use
    pbar.wandb_callback = callback
    
    return pbar


def log_tqdm_progress(pbar, n=1):
    """Log progress from a tqdm bar to wandb."""
    if hasattr(pbar, 'wandb_callback') and pbar.wandb_callback:
        pbar.wandb_callback(n)


def init_wandb(project: str, entity: str | None, config: dict, run_name: str | None = None):
    """Initialize a wandb run and log a config table."""
    run = wandb.init(project=project, entity=entity, config=config, name=run_name)
    # Log config as a table for readability
    config_table = wandb.Table(columns=["Parameter", "Value"])
    for key, value in config.items():
        config_table.add_data(key, str(value))
    wandb.log({"configuration": config_table})
    return run


def log_metrics(split_name: str, metrics: dict, step: int | None = None):
    """Log a flat dict of metrics with split prefix. Accepts already-prefixed keys too."""
    if not wandb.run:
        return
    log_dict = {}
    
    for key, value in metrics.items():
        # If key already has split prefix, keep it, else prefix
        if key.startswith(f"{split_name}/"):
            log_dict[key] = value
        else:
            log_dict[f"{split_name}/{key}"] = value

    # Log with the step
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


def log_confusion_matrix(split_name: str, y_true: list[int], y_pred: list[int], class_names: list[str]):
    if not wandb.run:
        return
    wandb.log({
        f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    })


def log_line_series(name: str, xs: list[int], ys_series: list[list[float]], keys: list[str], title: str, xname: str = "Epoch"):
    if not wandb.run:
        return
    wandb.log({
        name: wandb.plot.line_series(
            xs=xs,
            ys=ys_series,
            keys=keys,
            title=title,
            xname=xname,
        )
    })


def finish():
    if wandb.run:
        wandb.finish()


