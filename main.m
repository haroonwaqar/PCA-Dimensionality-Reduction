clc; clear; close all;

% SETUP & DATA LOADING
num_samples = 6000;
[X_original, labels] = load_mnist('dataset/train-images.idx3-ubyte','dataset/train-labels.idx1-ubyte',num_samples);

% TASK 1: ADD NOISE (SNR = 13 dB)
snr_db       = 13;
signal_power = mean(X_original(:).^2);
noise_power  = signal_power / (10^(snr_db/10));
noise_matrix = sqrt(noise_power) * randn(size(X_original));
X_noisy      = X_original + noise_matrix;

% TASK 2(a): SMALL DATASET VERIFICATION
X_test = [2 3; 3 4; 4 5; 5 6];
[Z_test, var_test, ~] = PCA_SVD(X_test, 2);
disp('--- SMALL DATASET PCA VERIFICATION ---');
disp('Projected Data (Z):');   disp(Z_test);
disp('Explained Variance:');   disp(var_test);

% TASK 2(b): FULL PCA — ALL COMPONENTS
% (needed for variance curve and dynamic threshold)

% max_k = min(samples, features) — gives the maximum
% number of non-zero principal components possible
max_k = min(size(X_original));

[~, var_orig_full,  ~] = PCA_SVD(X_original, max_k);
[~, var_noisy_full, ~] = PCA_SVD(X_noisy,    max_k);

% cumsum tells you what fraction of variance is captured by the top 100 components combined.
cum_orig  = cumsum(var_orig_full);
cum_noisy = cumsum(var_noisy_full);

% DYNAMIC THRESHOLD — Find k automatically
% This answers: "how many features do we need to keep?"

thresholds = [0.90, 0.95, 0.99];
fprintf('\n--- DYNAMIC THRESHOLD ANALYSIS ---\n');
for t = thresholds
    k_o = find(cum_orig  >= t, 1, 'first');
    k_n = find(cum_noisy >= t, 1, 'first');
    fprintf('%.0f%% variance — Original: k=%d | Noisy: k=%d | Difference: %d components\n', ...
        t*100, k_o, k_n, k_n - k_o);
end

% Store the 95% threshold k values for use in reconstruction + plots
k_orig_95  = find(cum_orig  >= 0.95, 1, 'first');
k_noisy_95 = find(cum_noisy >= 0.95, 1, 'first');
fprintf('\nUsing 95%% threshold: k_original=%d, k_noisy=%d\n', k_orig_95, k_noisy_95);

% RECONSTRUCTION ERROR vs k  (Dynamic accuracy analysis)
% This answers: "what is the effect of removing features?"

k_values     = [1, 5, 10, 20, 50, 100, 150, 200, 300, 500];
errors_orig  = zeros(size(k_values));
errors_noisy = zeros(size(k_values));

fprintf('\n--- RECONSTRUCTION ERROR vs k ---\n');
for i = 1:length(k_values)
    k_i = k_values(i);

    % --- Original ---
    [Z_o, ~, V_o]  = PCA_SVD(X_original, k_i);
    mu_o           = mean(X_original);
    X_rec_o        = Z_o * V_o' + mu_o;           % reconstruct
    errors_orig(i) = mean((X_original(:) - X_rec_o(:)).^2);

    % --- Noisy ---
    [Z_n, ~, V_n]   = PCA_SVD(X_noisy, k_i);
    mu_n            = mean(X_noisy);
    X_rec_n         = Z_n * V_n' + mu_n;
    errors_noisy(i) = mean((X_noisy(:) - X_rec_n(:)).^2);

    fprintf('k=%3d | Original MSE: %.5f | Noisy MSE: %.5f\n', ...
        k_i, errors_orig(i), errors_noisy(i));
end

% RECONSTRUCTION AT 95% THRESHOLD k
% This is the "recommended" k — not arbitrary

% Original at its own 95% k
[Z_o95, ~, V_o95] = PCA_SVD(X_original, k_orig_95);
mu_o95            = mean(X_original);
X_rec_o95         = Z_o95 * V_o95' + mu_o95;
err_orig_95       = mean((X_original(:) - X_rec_o95(:)).^2);

% Noisy at its own 95% k
[Z_n95, ~, V_n95] = PCA_SVD(X_noisy, k_noisy_95);
mu_n95            = mean(X_noisy);
X_rec_n95         = Z_n95 * V_n95' + mu_n95;
err_noisy_95      = mean((X_noisy(:) - X_rec_n95(:)).^2);

fprintf('\n--- RECONSTRUCTION AT 95%% THRESHOLD ---\n');
fprintf('Original (k=%d): MSE = %.5f\n', k_orig_95,  err_orig_95);
fprintf('Noisy    (k=%d): MSE = %.5f\n', k_noisy_95, err_noisy_95);

% PLOT 1: Cumulative Explained Variance + Threshold Lines

figure('Name', 'Explained Variance Analysis');
plot(cum_orig,  'b-',  'LineWidth', 2); hold on;
plot(cum_noisy, 'r--', 'LineWidth', 2);

% Threshold lines
yline(0.90, 'k:', '90%',  'LineWidth', 1.2);
yline(0.95, 'k-', '95%',  'LineWidth', 1.5);
yline(0.99, 'k-.','99%',  'LineWidth', 1.2);

% Mark where original and noisy cross 95%
xline(k_orig_95,  'b:', sprintf('k=%d (Orig)',  k_orig_95),  'LineWidth', 1.2);
xline(k_noisy_95, 'r:', sprintf('k=%d (Noisy)', k_noisy_95), 'LineWidth', 1.2);

legend('Original Dataset', 'Noisy Dataset (SNR 13)', 'Location', 'southeast');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('PCA Performance: Original vs Noisy MNIST');
grid on;

% PLOT 2: Reconstruction Error vs k

figure('Name', 'Reconstruction Error vs k');
plot(k_values, errors_orig,  'b-o', 'LineWidth', 2); hold on;
plot(k_values, errors_noisy, 'r--o','LineWidth', 2);

% Mark the 95% threshold k on this plot too
% Find nearest k_value to k_orig_95 for annotation
xline(k_orig_95,  'b:', sprintf('k=%d (Orig 95%%)',  k_orig_95));
xline(k_noisy_95, 'r:', sprintf('k=%d (Noisy 95%%)', k_noisy_95));

legend('Original MNIST', 'Noisy MNIST (SNR 13)', 'Location', 'best');
xlabel('Number of Principal Components (k)');
ylabel('Reconstruction MSE');
title('Effect of Dimensionality Reduction on Reconstruction Accuracy');
grid on;

% PLOT 3: 2D Visualization — always use k=2 for 2D scatter

[Z_2d_orig,  ~, ~] = PCA_SVD(X_original, 2);
[Z_2d_noisy, ~, ~] = PCA_SVD(X_noisy,    2);

figure('Name', '2D PCA Projections');
subplot(1,2,1);
gscatter(Z_2d_orig(:,1),  Z_2d_orig(:,2),  labels);
title('Original MNIST (2D)'); 
grid on;

subplot(1,2,2);
gscatter(Z_2d_noisy(:,1), Z_2d_noisy(:,2), labels);
title('Noisy MNIST (2D)'); 
grid on;

% PLOT 4: 3D Visualization — always use k=3 for 3D scatter

[Z_3d_orig,  ~, ~] = PCA_SVD(X_original, 3);
[Z_3d_noisy, ~, ~] = PCA_SVD(X_noisy,    3);

figure('Name', '3D PCA Projections');
subplot(1,2,1);
scatter3(Z_3d_orig(:,1),  Z_3d_orig(:,2),  Z_3d_orig(:,3),  15, labels, 'filled');
title('Original MNIST (3D)');
xlabel('PC 1'); 
ylabel('PC 2'); 
zlabel('PC 3'); 
grid on;

subplot(1,2,2);
scatter3(Z_3d_noisy(:,1), Z_3d_noisy(:,2), Z_3d_noisy(:,3), 15, labels, 'filled');
title('Noisy MNIST (3D)');
xlabel('PC 1'); 
ylabel('PC 2'); 
zlabel('PC 3'); 
grid on;