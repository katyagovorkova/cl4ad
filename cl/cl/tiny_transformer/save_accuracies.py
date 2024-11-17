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
    input = [76.0497291, 74.91534443, 79.47223297, 76.14164087, 73.58504257]
    mean_with_uncertainty(input)
if __name__=='__main__':
    main()
