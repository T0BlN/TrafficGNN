import matplotlib.pyplot as plt

def visualize_results(pred_etas, actual_etas):
    plt.scatter(actual_etas, pred_etas)
    plt.xlabel("Actual ETA")
    plt.ylabel("Predicted ETA")
    plt.title("Actual vs. Predicted ETA")
    plt.show()