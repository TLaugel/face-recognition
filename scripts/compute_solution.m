function [y] = compute_solution(t, centroids, nodes_to_centroids_map, centroids_to_nodes_map, labels)
        gamma_g = 1e-3;
        num_centroids = size(centroids,1);


        %compute the similarity matrix on the centroids
        %qW
        W = build_faces_graph('knn', 5, centroids, 2);



        %compute the multiplicities
        %v = zeros(length(labels),1);
        v = zeros(length(centroids_to_nodes_map),1);
        
        for i = 1:length(centroids_to_nodes_map)
            %v(----------(i)) = sum(---------------- == -----------------(i));
            v(i) = sum(nodes_to_centroids_map == centroids_to_nodes_map(i));
        end
        v = diag(v);
        
        % compute the labels of the nodes that are currently a
        % representative for a centroid
        sublabs = labels(centroids_to_nodes_map);

        %compute the normalized laplacian

        %qW = ;
        %sqrtD = ;
        %qL = ;
        %regL = ;

        Yl = [sublabs(sublabs ~= 0) == -1, sublabs(sublabs ~= 0) == 1];

        W2 = v*W*v;
        D = diag(sum(W2, 2));
        L = D - W2;
        
        all_idx = 1:num_centroids;
        l_idx = all_idx(sublabs ~= 0);
        u_idx = all_idx(sublabs == 0);
        
        % compute the SHFS solution

   %     Wul = ;
  %      Luu = ;
 %       Vuu = ;
%        Yu = full(  );
        Wul = W2(u_idx,l_idx);
        Luu = L(u_idx,u_idx);
        %Vuu = ;
        %Yu = full(  );
        %u_idx
        %L
        %Luu
        Yu = inv(Luu) * (Wul * Yl) ;

        %compute the labels
        y = sublabs;
%        y(sublabs == 0) = ;% +1 -1 labels
        [~, label_1_2] = max(Yu, [], 2);
        y(sublabs == 0) = 2*label_1_2 - 3;% +1 -1 labels

        
        
        fprintf('label %d confidence %.5f opposite %.5f time %d\n',y(find(centroids_to_nodes_map == t)),Yu(find(centroids_to_nodes_map(sublabs == 0) == t),1), Yu(find(centroids_to_nodes_map(sublabs == 0) == t),2),t);

        y = y(find(centroids_to_nodes_map == t));
