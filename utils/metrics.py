import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(classical_history, hybrid_history, title="Classical vs Hybrid Performance"):
    """
    Plots accuracy and loss comparison.
    """
    epochs = range(1, len(classical_history['acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, classical_history['acc'], 'b-o', label='Classical CNN')
    plt.plot(epochs, hybrid_history['acc'], 'r-s', label='Hybrid QNN')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 2. Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, classical_history['loss'], 'b-o', label='Classical CNN')
    plt.plot(epochs, hybrid_history['loss'], 'r-s', label='Hybrid QNN')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Saved evaluation plot to evaluation_results.png")
    plt.show()

def print_research_metrics(classical_stats, hybrid_stats):
    """
    Prints comparative research metrics.
    """
    print("\n" + "="*40)
    print("      QUANTUM RESEARCH METRICS")
    print("="*40)
    print(f"{'Metric':<25} | {'Classical':<10} | {'Hybrid':<10}")
    print("-" * 50)
    
    for key in classical_stats:
        print(f"{key:<25} | {classical_stats[key]:<10} | {hybrid_stats[key]:<10}")
    print("="*40)
