function X = RM_GAP(cost, Q, N)
%% this function obtain one feasible assignment for the current stage
    % @ cost: denotes the matrix of each user's cost, consisting of M shortest paths
    % @ Q: denotes the capacity of servers
    % @ N: denotes the number of users
    % @ X: a 0-1 matrix whose size is M*N (actually, there is no need to keep the decision varibale Y)

X = zeros(N, M);
cur_Q = Q;
Users = ones(1, N);

while sum(Users) ~= 0
    % malloc memory for difference & index (of server with minimum cost) between 1st and 2nd lowest cost, respectively
    delta_cost = zeros(N, 1);
    idx = zeros(N, 1);

    % if one server cannot process any request for users, set the cost as inf for these users
    [~, full_server] = find(cur_Q == 0);
    cost(:, full_server) = inf;

    % choose from feasible set of servers (servers who still have free computation source)
    for i = 1: N
        % this user has found his best placement, skip it
        if (Users(i) == 0)
            continue;
        end
        [min_cost, min_idx] = min(cost(i, :));
        cost(i, min_idx) = inf;
        delta_cost(i) = min(cost(i, :)) - min_cost;
    end
    [~, chosen_i] = min(delta_cost);
    X(chosen_i, idx(chosen_i)) = 1;

    % make preparations for next iteration
    Users(chosen_i) = 0;
    cur_Q(idx(chosen_i)) = cur_Q(idx(chosen_i)) - 1;
end

end