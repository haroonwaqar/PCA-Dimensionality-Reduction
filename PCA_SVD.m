function [Z, explained_variance, V_k] = PCA_SVD(X, k)
    % PCA_SVD: Performs Principal Component Analysis.
    % Inputs:
    %   X - Data matrix (N samples x D features)
    %   k - Number of principal components to keep
    % Outputs:
    %   Z - Projected data in lower-dimensional space (N x k)
    %   explained_variance - Proportion of variance explained by each component
    %   V_k - Top k principal components (eigenvectors)

    N = size(X, 1);

    % Step 1: Mean Centering
    % We subtract the mean of each feature to center the data around the origin.
    mu = mean(X);
    X_centered = X - mu;

    % Step 2: Singular Value Decomposition (SVD)
    % Using 'econ' computes only the necessary components, saving memory.
    [~, S, V] = svd(X_centered, 'econ');

    % Step 3: Extract the top 'k' principal components
    V_k = V(:, 1:k);

    % Step 4: Project the data
    % Transform the high-dimensional data into the new k-dimensional space.
    Z = X_centered * V_k;

    % Step 5: Calculate Explained Variance
    % Eigenvalues are the square of singular values divided by (N-1)
    eigenvalues = diag(S).^2 / (N - 1);
    explained_variance = eigenvalues / sum(eigenvalues);
end