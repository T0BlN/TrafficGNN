import matplotlib.pyplot as plt

#code to show the prediction vs actual ETA plot

def visualize_results(pred, actual, test_mask):
    pred = pred[test_mask].cpu().numpy()
    actual = actual[test_mask].cpu().numpy()

    plt.figure()
    plt.scatter(actual, pred)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color="red", linestyle="--")
    plt.xlabel("Actual ETA")
    plt.ylabel("Predicted ETA")
    plt.title("Actual vs. Predicted ETA")
    plt.show()

    residuals = pred - actual
    plt.figure()
    plt.scatter(actual, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual ETA")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residuals of ETA Predictions")
    plt.show()