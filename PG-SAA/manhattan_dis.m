function dis = manhattan_dis(point1, point2)
%% return the manhattan distance between point1 and point2
% point1 and point2 are both two-dimentional vectors
dis = abs(point1(1) - point2(1)) + abs(point1(2) - point2(2));
    
end