function dist = face_similarity_function(X)
% dist = face_similarity_function(X)
% Computes the distance matrix for a given dataset of samples.
% Follows the weighted norm formula on the handout
% X: is an n x m matrix of m-dimensional samples
% The return value dist is an n x n distance matrix
    EXTR_FRAME_SIZE = 96;
    X = double(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% build the EXTR_FRAME_SIZE  x EXTR_FRAME_SIZE matrix  %%%%%%
%%%%%% of weights W following the formula on the handout   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n = EXTR_FRAME_SIZE;
    
    a = ((1:n) - n).^2;
    b = repmat(a', 1, n) + 0.5 * repmat(a, n, 1);
    W = ones(n) - 1/(1.5 * n^2) * b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% use similarity function to build full graph (distances)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    W = sqrt(max(min(W,0.8),0.4)*(1/0.4) - 1);

    W = repmat(W(:)',size(X,1),1);

    dist = squareform(pdist(X.*W,'euclidean'));
    dist = min(dist, squareform(pdist((X -repmat(mean(X,2),1,size(X,2))).*W,'euclidean')));
    dist = min(dist, squareform(pdist((X./repmat(mean(X,2),1,size(X,2))).*W,'euclidean')));
