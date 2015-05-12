function [centroids, nodes_to_centroids_map, centroids_to_nodes_map, taboo, R] = update_centroids(centroids, max_num_centroids, nodes_to_centroids_map, centroids_to_nodes_map, taboo, R, num_labels, face, t)

      gamma_g = 1e-3;

      if (size(centroids,1) == max_num_centroids)
        % build distance matrix d_t
        [~,dt] = build_faces_graph('knn', 5, centroids, 2);

        %taboo nodes and self loops are infinitely distant, so they do not get merged
        dt(:, taboo) = 1e6;
        dt = dt + 1e6 * diag(ones(1, max_num_centroids));
        
        % find the position of the two closest vertices
%        dt(:)
 %       repv = ;% repv is the row i of the edge found
  %      addv = ;% addv is the column j of the edge found
        [repv, addv, dx] = find(dt==min(dt(:)));
        repv = repv(1);
        addv = addv(1);
        
        % if repv is not taboo and is not part of the labeled nodes
        % if addv represents more nodes then repv, swap addv and repv
        % so that the larger centroid gets larger
        %if ((~taboo(repv)) && (repv > num_labels) && ...
        if ((~taboo(repv)) && (sum(nodes_to_centroids_map == centroids_to_nodes_map(addv)) > sum(nodes_to_centroids_map == centroids_to_nodes_map(repv))))
          auxv = repv;
          repv = addv;
          addv = auxv;
        end
        
        size(R)
        size(dx)
        
        if (R == 0)
          % initialize R
          R = dx;
          taboo = [true(1, num_labels), false(1, max_num_centroids - num_labels)];
          fprintf('R = %.6f\n', R);
          fprintf('  %i quantization vertices\n', length(centroids_to_nodes_map));
        elseif (dx > R)
          % double R
          R = 1.5 * R;
          taboo = [true(1, num_labels), false(1, max_num_centroids - num_labels)];
          fprintf('R = %.6f\n', R);
          fprintf('  %i quantization vertices\n', length(centroids_to_nodes_map));
        end
        
        % update data structures
        % find all nodes associated with the addv centroid, and replace their
        % nodes_to_centroids_mapping to the new nodes that represents repv
        %nodes_to_centroids_map( ---------------- == -----------------(addv)) = ------------------(repv);
        nodes_to_centroids_map( nodes_to_centroids_map == centroids_to_nodes_map(addv)) = centroids_to_nodes_map(repv);

        % we don't want to instantly remerge something we already merged
        taboo(repv) = true;

        % the centroids_to_nodes_map of addv is replaced with the last node added
        % and the data in the centroids variable gets updated accordingly
        %-----------------------(------) = t;
        %---------(------,:) = face;
        centroids_to_nodes_map(addv) = t;
        centroids(addv,:) = face;
        
        % the new centroid point to himself, since he is the only node represented
        %----------------------(t) = t;
        nodes_to_centroids_map(t) = t;
      else

        % in the beginning we don't worry about repartitioning the centroids,
        % and just add them

        centroids_to_nodes_map = [centroids_to_nodes_map, t];
        nodes_to_centroids_map(t) = t;
        taboo(t) = false;
        centroids = [centroids;face];
        centroids = centroids;
      end
