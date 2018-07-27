clc, clear
%% ====================== Simulation of Parallel Greedy SAA-Based Application Placemnet Algorithm ====================

%% global variables
M = 5;                      % number of MEC servers
N = 50;                     % number of mobile users
T = 10;                     % number of time slots defined
H = 30;                     % number of samples (potential optimal solutions)
L = 100;                    % number of scenarios (sampling)
L_test = 1000;              % number of test scenarios (L_test >> L)
q = 8;                      % maximum number of containers for servers (for uniform distribution)

WIDTH = 100; HEIGHT = 100;                              % the restricted positions of users and servers
server_location = round(unifrnd(1, WIDTH, M, 2));       % the servers' location

gamma = round(unifrnd(1, 100, M, 1));                   % each server's computation cost
Q = round(unifrnd(1, q, M, 1));                         % each server's capacity (number of containers)
eta1  = 10; eta2 = 10;      % eta1 & eta2 represents the constant factors of communication cost and relocation cost, respectively

optimal_X = zeros(T, N);                                % the decision variable for each user at each time slot

%% the main part of PG-SAA
for t = 1: T
    % generate the locations of N users (it's certain at the begining of tth time slot)
    % we use s_t to denote the scenario
    s_t = round(unifrnd(1, WIDTH, N, 2));

    % malloc memory for the potimal X of H samples in tth time slot (ne need to store Y)
    X_t = zeros(N, M*H);
    
    %% =========================================== training ============================================
    % for each sample (k), we execute the same procedure
    for k = 1: H
        % generate L future scenarios of future time slots, which means we have (T-t)*L scenarios here
        % in order to simplified the problem, we add scenario of tth time slot to the variable
        scenarios = zeros(N*L, 2*(T-t)+2);
        scenarios(1:N*L, 1:2) = repmat(s_t, L, 1);
        for pointer_l = 0: (L-1)
            for pointer_t = 0: (T-t-1)
                scenarios(pointer_l*N+1: pointer_l*N+N, pointer_t*2+3: pointer_t*2+4) = ...
                round(unifrnd(1, WIDTH, N, 2));
            end
        end

        % malloc memory to store the M costs of each potential chosen server for each user
        cost = zeros(N, M);

        % for each user (i), generate L graphs (l) to simulate the potential future scenarios
        for i = 1: N
            % each user has M shortest path from server #1, #2, ..., #M, respectively
            for potential_chosen_server = 1: M
                % for each shoertest path, evaluate the chosen_server L times and get the average cost of L minimum costs
                % as the estimate cost of this decision (i.e., choosing the potential_chosen_server)

                % malloc memory to store the cost of the shortest path in L samplings
                omega_i_pcs = zeros(1, L);
                for pointer_l = 1: L
                    % to create the graph means to generate the adjcent matrix
                    % there are M*(T-t+1)+1 nodes in one graph (one for dummy vertix D)
                    G = zeros(M*(T-t+1)+1, M*(T-t+1)+1);
                    for pointer_t = t: (T-1)
                        for j1 = 1: M
                            for j2 = 1: M
                                % set the edge's value from S_{j1}^{pointer_t} to S_{j2}^{pointer_t+1}
                                % (it's unreachable from S_{j2}^{pointer_t+1} to S_{j1}^{pointer_t}!)
                                G(M*(pointer_t-t)+j1, M*(pointer_t+1-t)+j2) = ...
                                % computation cost of S_{j1}^{pointer_t}
                                gamma(j1) + ...
                                % communication cost between S_{j1}^{pointer_t} and U_i
                                eta1*manhattan_dis(scenarios((pointer_l-1)*N+i, ...
                                (pointer_t-t)*2+1:(pointer_t-t)*2+2), server_location(j1,:)) + ...
                                % relocation cost between S_{j1}^{pointer_t} and S_{j2}^{pointer_t+1}
                                % apparently, if j1 == j2, relocation cost is zero
                                eta2*manhattan_dis(server_location(j1,:), server_location(j2,:));
                            end
                        end
                    end
                    % set the unreachable edge's value as inf
                    G(find(G==0)) = inf;
                    
                    % use Dijkstra's algorithm to obtain the shortset path from potential_chosen_server to dummy vertix D
                    % ============ this part will be updated later ===========
                    % after this operation, we get the cost of the shortest path, omega

                    % store the cost of shortest path in this sampling
                    omega_i_pcs(pointer_l) = omega;
                end
                cost(i, potential_chosen_server) = mean(omega_i_pcs);
            end
        end

        % for now, we have obtained the placement matrix W_k (i.e., variable cost) for kth sample
        % (the row index of W_k denotes the index of the user, 
        %  the column index of W_k denotes the index of the potential chosen server)
        % time to call subroutinue RM-GAP to get the optimal placement for each user
        X_k_t = RM_GAP(cost, Q, N);

        % preserve the result of kth sample
        X_t(:, (k-1)*M+1: k*M) = X_k_t;
    end

    %% =========================================== testing ============================================
    % malloc memory for each samples' estimated cost for each user and then find the optimal (mimimum) one, that's the final decision!
    cost_for_all_samples = zeros(H, N);
    % test the decison of H samples and chooose the optimal one from them
    for k = 1: H
        % generate L_test future scenarios of future time slots, which means we have (T-t)*L_test scenarios here
        % in order to simplified the problem, we add scenario of tth time slot to the variable
        scenarios = zeros(N*L_test, 2*(T-t)+2);
        scenarios(1:N*L_test, 1:2) = repmat(s_t, L_test, 1);
        for pointer_l = 0: (L_test-1)
            for pointer_t = 0: (T-t-1)
                scenarios(pointer_l*N+1: pointer_l*N+N, pointer_t*2+3: pointer_t*2+4) = ...
                round(unifrnd(1, WIDTH, N, 2));
            end
        end

        % for each user (i), generate L_test graphs (l) to simulate the potential future scenarios
        for i = 1: N
            % each user has one potential chosen server whose index has been storn in X_t
            [~, potential_chosen_server] = find(X_t(i, (k-1)*M+1: k*M) == 1);
            % evaluate this potential chosen_server L_test times and get the average cost of L_test minimum costs
            % as the estimate cost of this decision (i.e., indeed choosing the potential_chosen_server)

            % malloc memory to store the cost of the shortest path in L_test samplings
            omega_i_pcs = zeros(1, L_test);
            for pointer_l = 1: L_test
                % to create the graph means to generate the adjcent matrix
                % there are M*(T-t+1)+1 nodes in one graph (one for dummy vertix D)
                G = zeros(M*(T-t+1)+1, M*(T-t+1)+1);
                for pointer_t = t: (T-1)
                    for j1 = 1: M
                        for j2 = 1: M
                            % set the edge's value from S_{j1}^{pointer_t} to S_{j2}^{pointer_t+1}
                            % (it's unreachable from S_{j2}^{pointer_t+1} to S_{j1}^{pointer_t}!)
                            G(M*(pointer_t-t)+j1, M*(pointer_t+1-t)+j2) = ...
                            % computation cost of S_{j1}^{pointer_t}
                            gamma(j1) + ...
                            % communication cost between S_{j1}^{pointer_t} and U_i
                            eta1*manhattan_dis(scenarios((pointer_l-1)*N+i, ...
                            (pointer_t-t)*2+1:(pointer_t-t)*2+2), server_location(j1,:)) + ...
                            % relocation cost between S_{j1}^{pointer_t} and S_{j2}^{pointer_t+1}
                            % apparently, if j1 == j2, relocation cost is zero
                            eta2*manhattan_dis(server_location(j1,:), server_location(j2,:));
                        end
                    end
                end
                % set the unreachable edge's value as inf
                G(find(G==0)) = inf;
                
                % use Dijkstra's algorithm to obtain the shortset path from potential_chosen_server to dummy vertix D
                % ============ this part will be updated later ===========
                % after this operation, we get the cost of the shortest path, omega

                % store the cost of shortest path in this sampling
                omega_i_pcs(pointer_l) = omega;
            end
            % store the estimated cost
            cost_for_all_samples(k, i) = mean(omega_i_pcs);
        end

        % for now, we have used L_test specimens to test H samples, time to choose the optimal one for each user!
        final_X = zeros(1, N);
        for i = 1: N
            [~, best_k] = find(cost_for_all_samples(:,i));
            % store the index of server j
            [~, final_X(i)] = find(X_t(i, (best_k-1)*M+1: best_k*M) == 1);
        end
    end

    % store the optimal decision for users at tth time slot
    optimal_X(t, :) = final_X;
end
