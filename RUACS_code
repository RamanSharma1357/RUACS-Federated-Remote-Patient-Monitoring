%% ================================================================
% RUACS: Reliability- and Uncertainty-Driven Adaptive Client Selection
% for Federated Remote Patient Monitoring
%
% IMPROVED / CORRECTED VERSION
%
% Main improvements:
%   1. Reliability is directly related to sensor noise.
%   2. Faulty sensors generate substantially noisier measurements.
%   3. Training/test classes are balanced consistently.
%   4. Prediction uncertainty uses Monte-Carlo sensor perturbations.
%   5. RUACS uses reliability + data quality + uncertainty.
%   6. A short random warm-up prevents selection from starting
%      before the global model has learned anything.
%   7. Score-based probabilistic selection improves client diversity.
%   8. RUACS and Random use exactly the same local training/FedAvg.
%   9. No artificial accuracy improvement is imposed.
%
% ================================================================

clear;
clc;
close all;

rng(42);

%% ================================================================
% 1. SIMULATION PARAMETERS
% ================================================================

N_clients  = 50;
N_rounds   = 100;

N_select   = 10;

N_train    = 200;
N_test     = 100;

N_features = 5;

% ---------------------------------------------------------------
% Fault probability
% ---------------------------------------------------------------

fault_probability = 0.20;

% ---------------------------------------------------------------
% Sensor noise
% ---------------------------------------------------------------

noise_min = 0.02;
noise_max = 0.25;

% Additional noise for faulty sensors
fault_noise_addition = 0.40;

% ---------------------------------------------------------------
% RUACS weights
% ---------------------------------------------------------------

w_reliability = 0.45;
w_uncertainty = 0.30;
w_dataquality = 0.25;

% ---------------------------------------------------------------
% Local logistic-regression training
% ---------------------------------------------------------------

learning_rate = 0.06;
local_epochs  = 6;

% ---------------------------------------------------------------
% Initial random warm-up
% ---------------------------------------------------------------

warmup_rounds = 5;

% ---------------------------------------------------------------
% Monte-Carlo uncertainty estimation
% ---------------------------------------------------------------

MC_samples = 8;

% Temperature controls diversity during score-based selection.
% Smaller value = stronger preference for high-score clients.
selection_temperature = 0.08;

%% ================================================================
% 2. GENERATE SENSOR CHARACTERISTICS
% ================================================================

noise_level   = zeros(N_clients,1);
faulty_sensor = zeros(N_clients,1);

% Physical measurement noise for:
% HR, SpO2, Temperature, Respiration, BP

sensor_noise_std = zeros(N_clients,N_features);

for i = 1:N_clients

    % Base sensor noise
    noise_level(i) = ...
        noise_min + ...
        (noise_max-noise_min)*rand;

    % Fault generation
    if rand < fault_probability

        faulty_sensor(i) = 1;

        % Faulty sensor has additional measurement noise
        noise_level(i) = ...
            noise_level(i) + fault_noise_addition;

    end

    % ------------------------------------------------------------
    % Client-specific measurement noise
    % ------------------------------------------------------------

    sensor_noise_std(i,:) = [ ...
        3.0  + 25*noise_level(i), ...
        0.7  + 5*noise_level(i), ...
        0.10 + 0.60*noise_level(i), ...
        1.0  + 5*noise_level(i), ...
        6.0  + 25*noise_level(i)];

end

%% ================================================================
% 3. SENSOR RELIABILITY
% ================================================================

% Reliability is directly derived from sensor noise.
%
% Higher noise -> lower reliability.
%
% Unlike the previous version, reliability is NOT min-max
% normalized across clients. This keeps the physical meaning.

reliability = ...
    exp(-2.2*noise_level);

reliability = ...
    min(max(reliability,0),1);

%% ================================================================
% 4. DATA QUALITY
% ================================================================

% Data quality is related to sensor quality but is not identical
% to reliability.

data_quality = zeros(N_clients,1);

for i = 1:N_clients

    quality_base = ...
        exp(-1.6*noise_level(i));

    % Small independent variation
    quality_variation = ...
        0.025*randn;

    data_quality(i) = ...
        quality_base + quality_variation;

end

data_quality = ...
    min(max(data_quality,0),1);

%% ================================================================
% 5. CLIENT PHYSIOLOGICAL BASELINES
% ================================================================

client_base = zeros(N_clients,N_features);

for i = 1:N_clients

    client_base(i,:) = [ ...
        70 + 1.5*randn, ...      % Heart rate
        97 + 0.15*randn, ...     % SpO2
        36.8 + 0.03*randn, ...   % Temperature
        16 + 0.4*randn, ...      % Respiration
        120 + 2.5*randn];        % Blood pressure

end

%% ================================================================
% 6. ABNORMAL PHYSIOLOGICAL CHANGES
% ================================================================

% Moderate abnormal shifts.
%
% These are large enough to create a meaningful classification
% problem, but not so large that accuracy becomes almost 100%.

abnormal_shift = [ ...
     4.5, ...      % HR
    -0.70, ...     % SpO2
     0.22, ...     % Temperature
     1.10, ...     % Respiration
     5.50];        % Blood pressure

%% ================================================================
% 7. GENERATE TRAINING DATA
% ================================================================

X_train = zeros(N_clients,N_train,N_features);
Y_train = zeros(N_clients,N_train);

for i = 1:N_clients

    % ------------------------------------------------------------
    % Exactly 30% abnormal observations.
    % This makes the experiment reproducible and avoids random
    % class imbalance between clients.
    % ------------------------------------------------------------

    y = zeros(N_train,1);

    positive_index = ...
        randperm(N_train,round(0.30*N_train));

    y(positive_index) = 1;

    Y_train(i,:) = y';

    for j = 1:N_train

        physiological_change = ...
            abnormal_shift*y(j);

        % --------------------------------------------------------
        % Measurement noise
        % --------------------------------------------------------

        measurement_noise = ...
            sensor_noise_std(i,:).*randn(1,N_features);

        % --------------------------------------------------------
        % Observed physiological measurements
        % --------------------------------------------------------

        X_train(i,j,:) = ...
            client_base(i,:) + ...
            physiological_change + ...
            measurement_noise;

    end

end

%% ================================================================
% 8. GENERATE INDEPENDENT TEST DATA
% ================================================================

X_test = zeros(N_clients,N_test,N_features);
Y_test = zeros(N_clients,N_test);

for i = 1:N_clients

    % Exactly 30% abnormal test observations

    y = zeros(N_test,1);

    positive_index = ...
        randperm(N_test,round(0.30*N_test));

    y(positive_index) = 1;

    Y_test(i,:) = y';

    for j = 1:N_test

        physiological_change = ...
            abnormal_shift*y(j);

        measurement_noise = ...
            sensor_noise_std(i,:).*randn(1,N_features);

        X_test(i,j,:) = ...
            client_base(i,:) + ...
            physiological_change + ...
            measurement_noise;

    end

end

%% ================================================================
% 9. FEATURE NORMALIZATION
% ================================================================

X_train_flat = ...
    reshape(X_train,[],N_features);

feature_mean = ...
    mean(X_train_flat,1);

feature_std = ...
    std(X_train_flat,[],1);

feature_std(feature_std < 1e-8) = 1;

% Normalize training

X_train_flat = ...
    (X_train_flat-feature_mean)./feature_std;

X_train = ...
    reshape(X_train_flat,...
    N_clients,N_train,N_features);

% Normalize test using training statistics

X_test_flat = ...
    reshape(X_test,[],N_features);

X_test_flat = ...
    (X_test_flat-feature_mean)./feature_std;

X_test = ...
    reshape(X_test_flat,...
    N_clients,N_test,N_features);

%% ================================================================
% 10. INITIAL GLOBAL MODEL
% ================================================================

% Logistic regression:
%
% P(y=1|x) = sigmoid(x*w+b)

global_w = ...
    zeros(N_features+1,1);

%% ================================================================
% 11. STORAGE VARIABLES
% ================================================================

accuracy_history = zeros(N_rounds,1);

balanced_accuracy_history = ...
    zeros(N_rounds,1);

f1_history = ...
    zeros(N_rounds,1);

random_accuracy = ...
    zeros(N_rounds,1);

random_balanced_accuracy = ...
    zeros(N_rounds,1);

random_f1 = ...
    zeros(N_rounds,1);

selected_uncertainty_history = ...
    zeros(N_rounds,1);

all_uncertainty_history = ...
    zeros(N_rounds,1);

selected_history = ...
    zeros(N_rounds,N_select);

score_history = ...
    zeros(N_rounds,N_clients);

selection_count = ...
    zeros(N_clients,1);

%% ================================================================
% 12. RUACS FEDERATED LEARNING
% ================================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf(' RUACS FEDERATED LEARNING\n');
fprintf('====================================================\n');

for round = 1:N_rounds

    %% ============================================================
    % 12.1 PREDICTION UNCERTAINTY
    %
    % IMPROVED:
    % Instead of relying only on entropy of the unperturbed model,
    % prediction uncertainty is estimated by repeatedly perturbing
    % each client's measurements according to its actual sensor
    % noise level.
    %
    % This gives a physically meaningful sensor-dependent
    % uncertainty estimate.
    % ============================================================

    uncertainty = zeros(N_clients,1);

    for i = 1:N_clients

        Xi = ...
            squeeze(X_train(i,:,:));

        uncertainty(i) = ...
            estimate_uncertainty( ...
            Xi,...
            global_w,...
            sensor_noise_std(i,:),...
            feature_std,...
            MC_samples);

    end

    uncertainty = ...
        min(max(uncertainty,0),1);

    %% ============================================================
    % 12.2 RUACS SCORE
    % ============================================================

    score = ...
        w_reliability*reliability + ...
        w_dataquality*data_quality - ...
        w_uncertainty*uncertainty;

    score_history(round,:) = ...
        score';

    %% ============================================================
    % 12.3 CLIENT SELECTION
    % ============================================================

    if round <= warmup_rounds

        % --------------------------------------------------------
        % Warm-up:
        % The model is initially untrained, therefore prediction
        % uncertainty is not yet informative.
        % --------------------------------------------------------

        selected_clients = ...
            randperm(N_clients,N_select);

    else

        % --------------------------------------------------------
        % Score-based probabilistic selection.
        %
        % This preserves preference for high RUACS scores while
        % preventing exactly the same 10 clients from being chosen
        % in every round.
        % --------------------------------------------------------

        selected_clients = ...
            sample_clients_by_score( ...
            score,...
            N_select,...
            selection_temperature);

    end

    selected_history(round,:) = ...
        selected_clients';

    %% ============================================================
    % 12.4 UPDATE SELECTION COUNTS
    % ============================================================

    for k = 1:N_select

        selection_count( ...
            selected_clients(k)) = ...
            selection_count( ...
            selected_clients(k)) + 1;

    end

    %% ============================================================
    % 12.5 UNCERTAINTY STATISTICS
    % ============================================================

    selected_uncertainty_history(round) = ...
        mean(uncertainty(selected_clients));

    all_uncertainty_history(round) = ...
        mean(uncertainty);

    %% ============================================================
    % 12.6 LOCAL TRAINING
    % ============================================================

    local_models = ...
        zeros(N_features+1,N_select);

    for k = 1:N_select

        client = ...
            selected_clients(k);

        Xi = ...
            squeeze(X_train(client,:,:));

        yi = ...
            Y_train(client,:)';

        Xi_bias = ...
            [Xi ones(N_train,1)];

        local_w = ...
            global_w;

        % --------------------------------------------------------
        % Local SGD
        % --------------------------------------------------------

        for epoch = 1:local_epochs

            logits = ...
                Xi_bias*local_w;

            probabilities = ...
                sigmoid(logits);

            error = ...
                probabilities-yi;

            gradient = ...
                (Xi_bias'*error)/N_train;

            local_w = ...
                local_w-learning_rate*gradient;

        end

        local_models(:,k) = ...
            local_w;

    end

    %% ============================================================
    % 12.7 FEDAVG
    % ============================================================

    global_w = ...
        mean(local_models,2);

    %% ============================================================
    % 12.8 GLOBAL TEST EVALUATION
    % ============================================================

    [accuracy_history(round),...
     balanced_accuracy_history(round),...
     f1_history(round)] = ...
        evaluate_global_model( ...
        global_w,...
        X_test,...
        Y_test,...
        N_clients,...
        N_test);

end

%% ================================================================
% 13. RANDOM CLIENT SELECTION BASELINE
% ================================================================

fprintf('\n');
fprintf('Running random-selection baseline...\n');

random_global_w = ...
    zeros(N_features+1,1);

for round = 1:N_rounds

    %% ------------------------------------------------------------
    % Random selection
    % ------------------------------------------------------------

    random_clients = ...
        randperm(N_clients,N_select);

    %% ------------------------------------------------------------
    % Local models
    % ------------------------------------------------------------

    random_models = ...
        zeros(N_features+1,N_select);

    for k = 1:N_select

        client = ...
            random_clients(k);

        Xi = ...
            squeeze(X_train(client,:,:));

        yi = ...
            Y_train(client,:)';

        Xi_bias = ...
            [Xi ones(N_train,1)];

        local_w = ...
            random_global_w;

        for epoch = 1:local_epochs

            logits = ...
                Xi_bias*local_w;

            probabilities = ...
                sigmoid(logits);

            error = ...
                probabilities-yi;

            gradient = ...
                (Xi_bias'*error)/N_train;

            local_w = ...
                local_w-learning_rate*gradient;

        end

        random_models(:,k) = ...
            local_w;

    end

    %% ------------------------------------------------------------
    % FedAvg
    % ------------------------------------------------------------

    random_global_w = ...
        mean(random_models,2);

    %% ------------------------------------------------------------
    % Test
    % ------------------------------------------------------------

    [random_accuracy(round),...
     random_balanced_accuracy(round),...
     random_f1(round)] = ...
        evaluate_global_model( ...
        random_global_w,...
        X_test,...
        Y_test,...
        N_clients,...
        N_test);

end

%% ================================================================
% 14. FINAL UNCERTAINTY
% ================================================================

final_uncertainty = ...
    zeros(N_clients,1);

for i = 1:N_clients

    Xi = ...
        squeeze(X_train(i,:,:));

    final_uncertainty(i) = ...
        estimate_uncertainty( ...
        Xi,...
        global_w,...
        sensor_noise_std(i,:),...
        feature_std,...
        MC_samples);

end

final_uncertainty = ...
    min(max(final_uncertainty,0),1);

%% ================================================================
% 15. RELIABILITY-UNCERTAINTY CORRELATION
% ================================================================

reliability_uncertainty_corr = ...
    corr(reliability,...
    final_uncertainty,...
    'Type','Pearson');

%% ================================================================
% 16. EXPECTED SELECTION FREQUENCY
% ================================================================

expected_selection = ...
    N_rounds*N_select/N_clients;

%% ================================================================
% 17. FINAL RESULTS
% ================================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FINAL RUACS RESULTS\n');
fprintf('====================================================\n');

fprintf('Clients                    : %d\n',N_clients);

fprintf('FL rounds                  : %d\n',N_rounds);

fprintf('Clients selected/round    : %d\n',N_select);

fprintf('Training samples/client   : %d\n',N_train);

fprintf('Test samples/client       : %d\n',N_test);

fprintf('Faulty sensors            : %d\n',sum(faulty_sensor));

fprintf('Fault percentage          : %.2f %%\n',...
    100*mean(faulty_sensor));

fprintf('\n');

fprintf('Average reliability       : %.3f\n',...
    mean(reliability));

fprintf('Average data quality      : %.3f\n',...
    mean(data_quality));

fprintf('\n');

fprintf('RUACS final accuracy      : %.2f %%\n',...
    100*accuracy_history(end));

fprintf('Random final accuracy     : %.2f %%\n',...
    100*random_accuracy(end));

fprintf('Accuracy improvement      : %.2f percentage points\n',...
    100*(accuracy_history(end)-random_accuracy(end)));

fprintf('\n');

fprintf('RUACS balanced accuracy   : %.2f %%\n',...
    100*balanced_accuracy_history(end));

fprintf('Random balanced accuracy  : %.2f %%\n',...
    100*random_balanced_accuracy(end));

fprintf('\n');

fprintf('RUACS F1-score            : %.3f\n',...
    f1_history(end));

fprintf('Random F1-score           : %.3f\n',...
    random_f1(end));

fprintf('\n');

fprintf('Reliability-Uncertainty correlation : %.3f\n',...
    reliability_uncertainty_corr);

fprintf('\n');

fprintf('Selection statistics:\n');

fprintf('Expected/client            : %.2f\n',...
    expected_selection);

fprintf('Minimum/client             : %d\n',...
    min(selection_count));

fprintf('Maximum/client             : %d\n',...
    max(selection_count));

fprintf('Mean/client                : %.2f\n',...
    mean(selection_count));

fprintf('\n');

fprintf('Selected-client uncertainty:\n');

fprintf('Initial round              : %.4f\n',...
    selected_uncertainty_history(1));

fprintf('Final round                : %.4f\n',...
    selected_uncertainty_history(end));

fprintf('\n');

fprintf('====================================================\n');

%% ================================================================
% 18. FIGURE 1
% SENSOR RELIABILITY
% ================================================================

figure('Name','Figure 1 - Sensor Reliability');

bar(1:N_clients,reliability);

xlabel('Healthcare Client');
ylabel('Reliability Score');

title('Sensor Reliability of Healthcare Clients');

xlim([0 N_clients+1]);
ylim([0 1]);

grid on;

%% ================================================================
% 19. FIGURE 2
% RELIABILITY VS PREDICTION UNCERTAINTY
% ================================================================

figure('Name','Figure 2 - Reliability vs Prediction Uncertainty');

scatter( ...
    reliability,...
    final_uncertainty,...
    55,...
    'filled');

hold on;

p = polyfit( ...
    reliability,...
    final_uncertainty,...
    1);

x_fit = ...
    linspace(min(reliability),...
    max(reliability),100);

y_fit = ...
    polyval(p,x_fit);

plot( ...
    x_fit,...
    y_fit,...
    'LineWidth',2);

xlabel('Sensor Reliability');

ylabel('Prediction Uncertainty');

title('Sensor Reliability vs Prediction Uncertainty');

xlim([0 1]);
ylim([0 1]);

legend( ...
    'Healthcare clients',...
    sprintf('Linear trend (r = %.2f)',...
    reliability_uncertainty_corr),...
    'Location','best');

grid on;

%% ================================================================
% 20. FIGURE 3
% RUACS VS RANDOM
% ================================================================

figure('Name','Figure 3 - RUACS vs Random Selection');

plot( ...
    1:N_rounds,...
    accuracy_history*100,...
    'LineWidth',2);

hold on;

plot( ...
    1:N_rounds,...
    random_accuracy*100,...
    '--',...
    'LineWidth',2);

xlabel('Federated Learning Round');

ylabel('Global Test Accuracy (%)');

title('RUACS vs Random Client Selection');

legend( ...
    'RUACS',...
    'Random',...
    'Location','southeast');

minimum_accuracy = ...
    min([accuracy_history;random_accuracy])*100;

maximum_accuracy = ...
    max([accuracy_history;random_accuracy])*100;

padding = max(2,...
    0.05*(maximum_accuracy-minimum_accuracy));

ylim([ ...
    max(0,minimum_accuracy-padding),...
    min(100,maximum_accuracy+padding)]);

grid on;

%% ================================================================
% 21. FIGURE 4
% SELECTED CLIENT UNCERTAINTY
% ================================================================

figure('Name','Figure 4 - Prediction Uncertainty');

plot( ...
    1:N_rounds,...
    selected_uncertainty_history,...
    'LineWidth',2);

hold on;

plot( ...
    1:N_rounds,...
    all_uncertainty_history,...
    '--',...
    'LineWidth',2);

xlabel('Federated Learning Round');

ylabel('Average Prediction Uncertainty');

title('Prediction Uncertainty of Selected Clients');

legend( ...
    'RUACS-selected clients',...
    'All clients',...
    'Location','northeast');

ylim([0 1]);

grid on;

%% ================================================================
% 22. FIGURE 5
% CLIENT SELECTION FREQUENCY
% ================================================================

figure('Name','Figure 5 - Client Selection Frequency');

bar( ...
    1:N_clients,...
    selection_count);

hold on;

plot( ...
    [1 N_clients],...
    [expected_selection expected_selection],...
    '--',...
    'LineWidth',2);

xlabel('Healthcare Client');

ylabel('Number of Selections');

title('RUACS Client Selection Frequency');

xlim([0 N_clients+1]);

ylim([0 max(N_rounds,...
    max(selection_count)+5)]);

legend( ...
    'Selection frequency',...
    'Expected average',...
    'Location','best');

grid on;

%% ================================================================
% 23. FIGURE 6
% NORMAL VS FAULTY SENSORS
% ================================================================

figure('Name','Figure 6 - Reliable and Faulty Sensors');

hold on;

normal_clients = ...
    find(faulty_sensor == 0);

fault_clients = ...
    find(faulty_sensor == 1);

plot( ...
    normal_clients,...
    reliability(normal_clients),...
    'o',...
    'MarkerSize',7,...
    'LineWidth',1.5);

plot( ...
    fault_clients,...
    reliability(fault_clients),...
    'x',...
    'MarkerSize',9,...
    'LineWidth',2);

xlabel('Healthcare Client');

ylabel('Reliability');

title('Reliable and Faulty Healthcare Sensors');

legend( ...
    'Normal sensor',...
    'Faulty sensor',...
    'Location','northeast');

xlim([0 N_clients+1]);
ylim([0 1]);

grid on;

%% ================================================================
% 24. FINAL INTERPRETATION CHECK
% ================================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf(' INTERPRETATION CHECK\n');
fprintf('====================================================\n');

if reliability_uncertainty_corr < 0

    fprintf(['PASS: Reliability and prediction uncertainty show ',...
        'a negative relationship.\n']);

else

    fprintf(['NOTE: Correlation is positive. This does not ',...
        'automatically mean the simulation is invalid.\n']);

end

if accuracy_history(end) > random_accuracy(end)

    fprintf(['PASS: RUACS final accuracy exceeds random ',...
        'selection.\n']);

else

    fprintf(['NOTE: Random selection has higher final accuracy. ',...
        'Check stochastic variation across repeated runs.\n']);

end

if selected_uncertainty_history(end) < ...
        all_uncertainty_history(end)

    fprintf(['PASS: RUACS-selected clients have lower final ',...
        'uncertainty than all clients.\n']);

else

    fprintf(['NOTE: Selected clients are not lower uncertainty ',...
        'than the complete population.\n']);

end

if min(selection_count) > 0

    fprintf(['PASS: Every client participated at least once ',...
        'in RUACS selection.\n']);

else

    fprintf(['NOTE: Some clients were never selected.\n']);

end

fprintf('====================================================\n');

%% ================================================================
% LOCAL FUNCTION 1
% SIGMOID
% ================================================================

function y = sigmoid(x)

    x = ...
        max(min(x,30),-30);

    y = ...
        1./(1+exp(-x));

end

%% ================================================================
% LOCAL FUNCTION 2
% MONTE-CARLO PREDICTION UNCERTAINTY
% ================================================================

function uncertainty = estimate_uncertainty( ...
    X,...
    model_w,...
    sensor_noise,...
    feature_std,...
    MC_samples)

    [N,F] = size(X);

    X_bias = ...
        [X ones(N,1)];

    % ------------------------------------------------------------
    % Baseline prediction
    % ------------------------------------------------------------

    base_probability = ...
        sigmoid(X_bias*model_w);

    base_probability = ...
        min(max(base_probability,1e-8),1-1e-8);

    base_entropy = ...
        -(base_probability.*log2(base_probability) + ...
        (1-base_probability).* ...
        log2(1-base_probability));

    base_entropy_mean = ...
        mean(base_entropy);

    % ------------------------------------------------------------
    % Convert physical sensor noise into normalized feature noise
    % ------------------------------------------------------------

    normalized_noise = ...
        sensor_noise./feature_std;

    normalized_noise = ...
        max(normalized_noise,0);

    % ------------------------------------------------------------
    % Monte-Carlo predictions
    % ------------------------------------------------------------

    prediction_matrix = ...
        zeros(N,MC_samples);

    for m = 1:MC_samples

        perturbation = ...
            randn(N,F).*normalized_noise;

        X_perturbed = ...
            X + perturbation;

        X_perturbed_bias = ...
            [X_perturbed ones(N,1)];

        p = ...
            sigmoid(X_perturbed_bias*model_w);

        p = ...
            min(max(p,1e-8),1-1e-8);

        prediction_matrix(:,m) = ...
            p;

    end

    % ------------------------------------------------------------
    % Prediction variability
    % ------------------------------------------------------------

    prediction_variance = ...
        mean(var(prediction_matrix,0,2));

    % Convert variance approximately to [0,1].
    %
    % Bernoulli probability variance has maximum 0.25.

    variability_score = ...
        min(4*prediction_variance,1);

    % ------------------------------------------------------------
    % Final uncertainty
    %
    % 40% predictive entropy
    % 60% prediction instability under sensor perturbation
    % ------------------------------------------------------------

    uncertainty = ...
        0.40*base_entropy_mean + ...
        0.60*variability_score;

    uncertainty = ...
        min(max(uncertainty,0),1);

end

%% ================================================================
% LOCAL FUNCTION 3
% SCORE-BASED CLIENT SAMPLING
% ================================================================

function selected = sample_clients_by_score( ...
    score,...
    N_select,...
    temperature)

    N = length(score);

    selected = zeros(N_select,1);

    available = true(N,1);

    for k = 1:N_select

        current_score = ...
            score;

        current_score(~available) = ...
            -Inf;

        maximum_score = ...
            max(current_score);

        scaled_score = ...
            (current_score-maximum_score) ...
            /temperature;

        probability = ...
            exp(scaled_score);

        probability(~available) = 0;

        total_probability = ...
            sum(probability);

        % Safety check

        if total_probability <= 0

            candidates = ...
                find(available);

            chosen = ...
                candidates(randi(length(candidates)));

        else

            probability = ...
                probability/total_probability;

            cumulative_probability = ...
                cumsum(probability);

            random_number = ...
                rand;

            chosen = ...
                find(cumulative_probability >= ...
                random_number,1,'first');

        end

        selected(k) = ...
            chosen;

        available(chosen) = false;

    end

end

%% ================================================================
% LOCAL FUNCTION 4
% GLOBAL MODEL EVALUATION
% ================================================================

function [accuracy,...
          balanced_accuracy,...
          f1_score] = ...
          evaluate_global_model( ...
          model_w,...
          X_test,...
          Y_test,...
          N_clients,...
          N_test)

    all_true = [];
    all_pred = [];

    for i = 1:N_clients

        Xi = ...
            squeeze(X_test(i,:,:));

        yi = ...
            Y_test(i,:)';

        Xi_bias = ...
            [Xi ones(N_test,1)];

        logits = ...
            Xi_bias*model_w;

        probabilities = ...
            sigmoid(logits);

        predictions = ...
            double(probabilities >= 0.5);

        all_true = ...
            [all_true;yi];

        all_pred = ...
            [all_pred;predictions];

    end

    %% ------------------------------------------------------------
    % Accuracy
    % ------------------------------------------------------------

    accuracy = ...
        mean(all_pred == all_true);

    %% ------------------------------------------------------------
    % Confusion matrix values
    % ------------------------------------------------------------

    TP = sum( ...
        (all_pred == 1) & ...
        (all_true == 1));

    TN = sum( ...
        (all_pred == 0) & ...
        (all_true == 0));

    FP = sum( ...
        (all_pred == 1) & ...
        (all_true == 0));

    P = ...
        sum(all_true == 1);

    N = ...
        sum(all_true == 0);

    %% ------------------------------------------------------------
    % Sensitivity
    % ------------------------------------------------------------

    sensitivity = ...
        TP/(P+eps);

    %% ------------------------------------------------------------
    % Specificity
    % ------------------------------------------------------------

    specificity = ...
        TN/(N+eps);

    %% ------------------------------------------------------------
    % Balanced accuracy
    % ------------------------------------------------------------

    balanced_accuracy = ...
        0.5*(sensitivity+specificity);

    %% ------------------------------------------------------------
    % Precision
    % ------------------------------------------------------------

    precision = ...
        TP/(TP+FP+eps);

    %% ------------------------------------------------------------
    % F1-score
    % ------------------------------------------------------------

    f1_score = ...
        2*precision*sensitivity/...
        (precision+sensitivity+eps);

end
