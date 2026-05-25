function [X, labels] = load_mnist(image_file, label_file, num_samples)
    % LOAD_MNIST: Reads IDX format MNIST files and normalizes the data.
    
    % Load Labels
    fid = fopen(label_file, 'r');
    fread(fid, 2, 'int32', 0, 'ieee-be'); % Skip magic number and count
    labels = fread(fid, num_samples, 'unsigned char');
    fclose(fid);
    
    % Load Images
    fid = fopen(image_file, 'r');
    fread(fid, 4, 'int32', 0, 'ieee-be'); % Skip magic, count, rows, cols
    images = fread(fid, num_samples * 28 * 28, 'unsigned char');
    fclose(fid);
    
    % Reshape and Normalize
    images = reshape(images, 28*28, num_samples)';
    X = double(images) / 255;
    labels = double(labels);
end