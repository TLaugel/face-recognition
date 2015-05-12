function [gray_face_vector] = extract_face_features(gray_face)
%function [gray_face] = extract_face_features(gray_face)
% Transforms a n x n image into into an 1 x n^2 feature vector
% gray_face: is an n x m image
% The return value W is an n x n graph matrix, while the distances
% matrix (if requested) contains the full matrix with all the distances
    EXTR_FRAME_SIZE = 96;

    %use cv.resize
    gray_face = cv.resize(gray_face, [EXTR_FRAME_SIZE, EXTR_FRAME_SIZE]);

    %use cv.equalizeHist to smoothen the picture
    gray_face = cv.equalizeHist(gray_face);
    gray_face = cv.GaussianBlur(gray_face,'KSize',[3,3],'SigmaX',0,'SigmaY',0);

    %necessary conversion from 8bit images
    gray_face = double(gray_face);

    %scale the data to [0,1]
    gray_face = gray_face./255;

    %force the cast to row vector
    gray_face_vector = gray_face(:)';
