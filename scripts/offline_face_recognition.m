function [accuracy] = offline_face_recognition(graph_type, graph_thresh)
% spectral_clustering
% a skeleton function to perform spectral clustering, needs to be completed
% graph_type, graph_thresh: in batch mode, control the graph construction,
%                           when missing simply plot the result
% ARI: return the quality of the clustering


%%%%%%%%%%%% importing the dataset

cc = cv.CascadeClassifier('haarcascade_frontalface_default.xml');

EXTR_FRAME_SIZE = 96;

X = zeros(100,EXTR_FRAME_SIZE^2);
Y = zeros(100,1);

for i = 0:9
    for j = 1:10
        im = imread(sprintf('data/10faces/%d/%02d.jpg',i,j));
        box = cc.detect(im);
        top_face.area = 0;
        frame.width = size(im,2);
        frame.height = size(im,1);
        for z = 1:length(box)
            rect.x = box{z}(1);
            rect.y = box{z}(2);
            rect.width = box{z}(3);
            rect.height = box{z}(4);
            face_area = rect.width*rect.height;
            if (face_area > top_face.area)
                top_face.area = face_area;
                top_face.x = box{z}(1);
                top_face.y = box{z}(2);
                top_face.width = box{z}(3);
                top_face.height = box{z}(4);
            end
        end
        gray_im = cv.cvtColor(im, 'BGR2GRAY');
        gray_face = gray_im( top_face.y:top_face.y + top_face.height, top_face.x:top_face.x + top_face.width);
        gray_face = extract_face_features(gray_face);
        X(sub2ind([10,10],i+1,j),:) = gray_face;
        Y(sub2ind([10,10],i+1,j),1) = i+1;
    end
end


%# show them in subplots
figure(1)
for i=1:20
    subplot(2,10,i);
    h = imshow(reshape(X(i,:),EXTR_FRAME_SIZE,EXTR_FRAME_SIZE), 'InitialMag',100, 'Border','tight');
    title(num2str(i))
end


plot_results = 0;

if nargin < 1

    plot_results = 1;

    %%%%%%%%%%%% the type of the graph to build and the respective threshold

    graph_type = 'knn';
    graph_thresh = 7; % the number of neighbours for the graph

    %graph_type = 'eps';
    %graph_thresh = 0.001; % the epsilon threshold

end


%%%%%%%%%%%% similarity options
similarity_options = [2]; % exponential_euclidean: sigma

%%%%%%%%%%%% automatically infer number of labels from samples
num_classes = length(unique(Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the build_faces_graph function to build the graph W  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = build_faces_graph(graph_type, graph_thresh, X, similarity_options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build the laplacian                                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = zeros(size(W));
for i=1:length(W)
    D(i,i) = sum(W(i,:));
end 
L = D - W ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select 4 random labels per person and reveal them            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_samples = size(Y, 1);
l_idx = [];
new_order = randperm(100);
Y_randomized = Y(new_order);
for face = 1:10
    l_sample = new_order(Y_randomized == face);
    l_idx = [l_idx l_sample(1:4)];
end
Y_l = Y(l_idx);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute hfs solution                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_idx = setdiff(1:num_samples, l_idx);

%Yl = zeros(length(l_idx),2);
Yl = zeros(length(l_idx),2);
for i = 1:10
    Yl(:,i) = Y_l==i;
end%    Yl(:,1) = Y_l==1;Yl(:,2) = Y_l==2;

Yu = inv(L(u_idx,u_idx)) * (W(u_idx,l_idx) * Yl) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the HFS solution       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,label] = max(Yu, [], 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final_label = Y;
final_label(u_idx) = label;

set(figure(), 'units', 'centimeters', 'pos', [0 0 20 10]);

h(1) = subplot(1,2,1);
imagesc(reshape(Y,10,10));
title('True labels')
   
h(2) = subplot(1,2,2);
imagesc(reshape(final_label,10,10));
title('HFS classification')

accuracy = mean(Y == final_label);

end
