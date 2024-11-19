import numpy as np

def mean_with_uncertainty(input):
    """Calculates the mean and standard deviation for an array of accuracies 
    using the corrected sample standard deviation"""
    mean = np.mean(input)
    N = len(input)
    #Sample standard deviation
    var = np.sum((input-mean)**2)/(len(input)-1)
    std = np.sqrt(var)
    mean_std = std/np.sqrt(N)
    print(f"Resulting mean and mean_std: {mean:.4f}+/-{mean_std:.4f}")
    return mean, mean_std


def main():
    input = [80.28492647,79.73829334,80.29943885,79.91244195,79.94388545]
    mean_with_uncertainty(input)
if __name__=='__main__':
    main()
    embedding = np.array([2,4,8,16,24,32])
    accuracy = np.array([29.9946,76.0328,79.2836,79.866,79.6783,80.0358])
    error=np.array([0,0.9768,0.2252,0.0904,0.2331,0.1104])
    np.savez("vicreg_acc_error.npz", embedding=embedding, accuracy=accuracy, error=error)
