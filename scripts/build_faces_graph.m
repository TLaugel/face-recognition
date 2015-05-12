function [W,distances] = build_faces_graph(graph_type, graph_thresh, X, gaussian_var)
% [W,distances] = build_faces_graph(graph_type, graph_thresh, X, gaussian_var)
% Computes the similarity matrix for a given dataset of samples.
% graph_type: knn or eps graph, as a string, controls the graph that
% the function will produce
% graph_thresh: controls the main parameter of the graph, the number
% of neighbours k for k-nn, and the threshold eps for epsilon graphs
% X: is an n x m matrix of m-dimensional samples
% similarity_options: is a vector containing options for the exponential
% RBF
% The return value W is an n x n graph matrix, while the distances
% matrix (if requested) contains the full matrix with all the distances


if nargin < 4
    error('build_similarity_graph: not enough arguments')
elseif nargin < 5
    similarity_options = [];
elseif nargin > 6
    error('build_similarity_graph: too many arguments')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% use similarity function to build full graph (distances)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

distances = face_similarity_function(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(X,1);

exp_distances = exp(-distances./(2*gaussian_var^2));

W = zeros(n,n);

if strcmp(graph_type,'knn') == 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%% compute a k-nn graph from the distances            %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [D,I] = sort(exp_distances,2,'descend');

    i_index = I(:, 1:graph_thresh);
    j_index = repmat([1:n]', 1,graph_thresh);
    z_values = D(:, 1:graph_thresh);

    W(sub2ind(size(W),j_index(:),i_index(:))) = z_values(:);
    W(sub2ind(size(W),i_index(:),j_index(:))) = z_values(:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(graph_type,'eps') == 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%% compute an epsilon graph from the distances        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    W(exp_distances >= graph_thresh) = exp_distances(exp_distances >= graph_thresh);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else

    error('build_similarity_graph: not a valid graph type')

end

