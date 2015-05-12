    function ossl_wrap(cap)
    %video enregistree: 
    %capvid = cv.VideoCapture('C:/Users/Thibault/Desktop/MVA/Graph/TP2/test.avi');
    
    %live hollywood
    
    
    
    cc = cv.CascadeClassifier('haarcascade_frontalface_default.xml');

    EXTR_FRAME_SIZE = 96;

    max_num_centroids = 100;

    figure();
    f = imshow(cv.resize(cap.read(),[640,480]));

    centroids = [];
    labels = [];
    for i = 1:10
        %1st user
        centroids = [centroids;extract_face_features(imread(sprintf('%s_%d.bmp','thibault laugel',i)))];
        labels = [labels;1];
        %2nd user
        centroids = [centroids;extract_face_features(imread(sprintf('%s_%d.bmp','raphael',i)))];
        labels = [labels;-1];
    end

    
    num_labels = size(labels,1);
    
    nodes_to_centroids_map = [1:20];
    centroids_to_nodes_map = [1:20];
    taboo = true(size(labels));
    R = 0;

    t = 17;
    while t < 1000
        im = cv.resize(cap.read(),[640,480]);
        box = cc.detect(im);
        frame.width = size(im,2);
        frame.height = size(im,1);

        top_face.idx = 0;
        top_face.dst = Inf;
        for i = 1:length(box)
            rect.x = box{i}(1);
            rect.y = box{i}(2);
            rect.width = box{i}(3);
            rect.height = box{i}(4);
            %im = cv.rectangle(im,box{i});
			%face_dist = norm([rect.x + rect.width / 2 - frame.width / 2, rect.y + rect.height / 2 - frame.height / 2]);
			%if (face_dist < top_face.dst)
				%top_face.idx = i;
				%top_face.dst = face_dist;
                %top_face.x = box{i}(1);
                %top_face.y = box{i}(2);
                %top_face.width = box{i}(3);
                %top_face.height = box{i}(4);
            %end
            gray_im = cv.cvtColor(im, 'BGR2GRAY');
            gray_face = gray_im( rect.y:rect.y + rect.height, rect.x:rect.x + rect.width);
            gray_face = extract_face_features(gray_face);

            [centroids, nodes_to_centroids_map, centroids_to_nodes_map, taboo, R] = update_centroids(centroids, max_num_centroids, nodes_to_centroids_map, centroids_to_nodes_map, taboo, R, num_labels, gray_face, t);
            labels = [labels;0];
            y = compute_solution(t, centroids, nodes_to_centroids_map, centroids_to_nodes_map, labels);
            if y ==1
                im = cv.rectangle(im,box{i},'Color',[255,0,0],'Thickness',2);
            else
                im = cv.rectangle(im,box{i},'Color',[0,255,0],'Thickness',2);
            end

            t = t+1;

        end
        set(f,'CData', im);
        drawnow()
        pause(1/30)



    end
