import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def parse_training_log(log_file):
    """Parse the training log file, focusing on step information."""
    steps = []
    losses = []
    lrs = []
    ms_steps = []
    samples_secs = []
    eval_steps = []
    eval_losses = []
    
    last_step = -1
    parsed_lines = 0
    total_lines = 0
    
    with open(log_file, 'r') as f:
        content = f.readlines()
        total_lines = len(content)
        
        for line in content:
            if "Step:" in line:
                # More flexible regex pattern to match step information
                match = re.search(r"Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.e-]+)\s*\|\s*ms/step:\s*([\d.]+)\s*\|\s*Samples/sec:\s*([\d.]+)", line)
                if match:
                    parsed_lines += 1
                    step, loss, lr, ms_step, samples_sec = match.groups()
                    step = int(step)
                    
                    # Only skip if exact same step (not if greater than last)
                    if step == last_step:
                        continue
                    
                    last_step = step
                    steps.append(step)
                    losses.append(float(loss))
                    lrs.append(float(lr))
                    ms_steps.append(float(ms_step))
                    samples_secs.append(float(samples_sec))
            elif "Evaluation loss:" in line:
                match = re.search(r"Evaluation loss:\s*([\d.]+)", line)
                if match and steps:  # Only if we have seen at least one step
                    eval_losses.append(float(match.group(1)))
                    eval_steps.append(last_step)  # Associate with the last seen step
    
    print(f"Total lines in log: {total_lines}")
    print(f"Matched training step lines: {parsed_lines}")
    print(f"Parsed {len(steps)} unique training steps (min: {min(steps) if steps else 'N/A'}, max: {max(steps) if steps else 'N/A'})")
    print(f"Parsed {len(eval_losses)} evaluation points")
    
    # Check for excessive data gaps
    if len(steps) > 1:
        step_diffs = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
        avg_step_diff = sum(step_diffs) / len(step_diffs)
        max_step_diff = max(step_diffs)
        print(f"Average step difference: {avg_step_diff}, Maximum step difference: {max_step_diff}")
    
    return {
        'train': (steps, losses, lrs, ms_steps, samples_secs),
        'eval': (eval_steps, eval_losses)
    }

def create_visualizations(data, output_dir):
    """Create and save visualization plots."""
    train_data = data['train']
    eval_data = data['eval']
    
    steps, losses, lrs, ms_steps, samples_secs = train_data
    eval_steps, eval_losses = eval_data
    
    # Sanity check for data
    print(f"Plotting data: {len(steps)} points (step range: {min(steps) if steps else 'N/A'}-{max(steps) if steps else 'N/A'})")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Training and Evaluation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', label='Training Loss', alpha=0.7)
    plt.plot(eval_steps, eval_losses, 'r-', label='Evaluation Loss', linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add logarithmic x-axis version
    plt.xscale('log')
    plt.tight_layout()
    loss_log_path = os.path.join(output_dir, "loss_log_scale.png")
    plt.savefig(loss_log_path, dpi=300)
    plt.xscale('linear')
    loss_path = os.path.join(output_dir, "loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    
    # 2. Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, 'g-')
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add logarithmic x-axis version
    plt.xscale('log')
    plt.tight_layout()
    lr_log_path = os.path.join(output_dir, "learning_rate_log_scale.png")
    plt.savefig(lr_log_path, dpi=300)
    plt.xscale('linear')
    lr_path = os.path.join(output_dir, "learning_rate.png")
    plt.savefig(lr_path, dpi=300)
    plt.close()
    
    # 3. Performance Metrics (ms/step and Samples/sec)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Milliseconds per Step
    ax1.plot(steps, ms_steps, 'm-')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("ms/step")
    ax1.set_title("Processing Time per Step")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Samples per Second
    ax2.plot(steps, samples_secs, 'c-')
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Samples/sec")
    ax2.set_title("Throughput")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    perf_path = os.path.join(output_dir, "performance_metrics.png")
    plt.savefig(perf_path, dpi=300)
    
    # Also create log scale version
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    perf_log_path = os.path.join(output_dir, "performance_metrics_log_scale.png")
    plt.savefig(perf_log_path, dpi=300)
    plt.close()
    
    # 4. Combined dashboard (all metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(steps, losses, 'b-', label='Training Loss')
    axes[0, 0].plot(eval_steps, eval_losses, 'r-', label='Evaluation Loss')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Evaluation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Learning Rate
    axes[0, 1].plot(steps, lrs, 'g-')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Learning Rate")
    axes[0, 1].set_title("Learning Rate Schedule")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Milliseconds per Step
    axes[1, 0].plot(steps, ms_steps, 'm-')
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("ms/step")
    axes[1, 0].set_title("Processing Time per Step")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Samples per Second
    axes[1, 1].plot(steps, samples_secs, 'c-')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Samples/sec")
    axes[1, 1].set_title("Throughput")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "training_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    
    # Create log scale version too
    for ax in axes.flatten():
        ax.set_xscale('log')
    dashboard_log_path = os.path.join(output_dir, "training_dashboard_log_scale.png")
    plt.savefig(dashboard_log_path, dpi=300)
    plt.close()
    
    return {
        'loss': loss_path,
        'loss_log': loss_log_path,
        'lr': lr_path,
        'lr_log': lr_log_path,
        'performance': perf_path,
        'performance_log': perf_log_path,
        'dashboard': dashboard_path,
        'dashboard_log': dashboard_log_path
    }

# Define input and output paths
input_dir = "/home/idl/neural_architecture_integration/outputs/byte_lm_final/logs"
output_dir = "/home/idl/neural_architecture_integration/outputs/byte_lm_final/vis"

# Get the log file
log_file = os.path.join(input_dir, "training_20250317_131607.txt")

# Parse the log file
data = parse_training_log(log_file)

# Create visualizations
plot_files = create_visualizations(data, output_dir)

print(f"Generated the following visualization files:")
for name, path in plot_files.items():
    print(f"- {name}: {path}")

