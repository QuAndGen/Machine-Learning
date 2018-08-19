from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

def missing_imput(data):
    # X is the complete data matrix
    # X_incomplete has the same values as X except a subset have been replace with NaN
      # Use 3 nearest rows which have a feature to fill in each row's missing features
    X_filled_knn = KNN(k=3).complete(X_incomplete)

    # matrix completion using convex optimization to find low-rank solution
    # that still matches observed values. Slow!
    X_filled_nnm = NuclearNormMinimization().complete(data)
    
    # Instead of solving the nuclear norm objective directly, instead
    # induce sparsity using singular value thresholding
    X_filled_softimpute = SoftImpute().complete(data)
    
    # print mean squared error for the three imputation methods above
    nnm_mse = ((data[missing_mask] - X[missing_mask]) ** 2).mean()
    print("Nuclear norm minimization MSE: %f" % nnm_mse)
    
    softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
    print("SoftImpute MSE: %f" % softImpute_mse)
    
    knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
    print("knnImpute MSE: %f" % knn_mse)
    #Algorithms
    #SimpleFill: Replaces missing entries with the mean or median of each column.
    #KNN: Nearest neighbor imputations which weights samples using the mean squared difference on features 
    #for which two rows both have observed data.
    #SoftImpute: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the 
    #softImpute package for R, which is based on Spectral Regularization Algorithms for 
    #Learning Large Incomplete Matrices by Mazumder et. al.
    #IterativeSVD: Matrix completion by iterative low-rank SVD decomposition. Should be similar to 
    #SVDimpute from Missing value estimation methods for DNA microarrays by Troyanskaya et. al.
    #MICE: Reimplementation of Multiple Imputation by Chained Equations.
    #MatrixFactorization: Direct factorization of the incomplete matrix into low-rank U and V, with 
    #an L1 sparsity penalty on the elements of U and an L2 penalty on the elements of V. Solved by 
    #gradient descent.
    #NuclearNormMinimization: Simple implementation of Exact Matrix Completion via Convex Optimization 
    #by Emmanuel Candes and Benjamin Recht using cvxpy. Too slow for large matrices.
    #BiScaler: Iterative estimation of row/column means and standard deviations to get doubly 
    #normalized matrix. Not guaranteed to converge but works well in practice. Taken from Matrix 
    #Completion and Low-Rank SVD via Fast Alternating Least Squares.
        
