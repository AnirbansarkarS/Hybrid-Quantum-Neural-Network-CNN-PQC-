from utils.metrics import plot_training_results, print_research_metrics

def main():
    print("--- Phase 4: Comparative Evaluation (Demo Data) ---")
    
    # In a real run, these would come from training logs
    # Using representative data for demonstration
    classical_history = {
        'acc': [85, 92, 94, 95, 96],
        'loss': [0.5, 0.3, 0.2, 0.15, 0.1]
    }
    
    hybrid_history = {
        'acc': [70, 82, 88, 91, 93], # Initially slower convergence
        'loss': [0.8, 0.5, 0.3, 0.2, 0.18]
    }
    
    classical_stats = {
        'Parameter Count': '25,482',
        'Training Time (1 epoch)': '1s',
        'Final Test Acc': '96.2%'
    }
    
    hybrid_stats = {
        'Parameter Count': '12 (Quantum) + CNN',
        'Training Time (1 epoch)': '120s (Simulated)',
        'Final Test Acc': '93.5%'
    }
    
    print_research_metrics(classical_stats, hybrid_stats)
    
    print("\nGenerating Comparison Plot...")
    # This will attempt to show a plot. If in a headless env, it saves to file.
    plot_training_results(classical_history, hybrid_history)

if __name__ == "__main__":
    main()
