%% Script for Prediction of PM2D5
clear; clc; close all;

%% Analysis Control

% Hyp. param. opt. (training) or making predictions for test data (test)
% predopt.stage = "training";
predopt.stage = "training_local";
% predopt.stage = "test";

% predopt.sampling = "random";
% predopt.sampling = "last_num_smpl";
predopt.sampling = "hybrid";
predopt.sampling_hybrid_past_ratio = 0.1;

% predopt.train_local_sizes = [1200*1.00, 1200*0.25]; % train test sampling sizes
% predopt.train_local_sizes = [2400*1.00, 2400*0.25]; % train test sampling sizes
predopt.train_local_sizes = [6000*1.00, 6000*0.25]; % train test sampling sizes

% Which type of prediction to make
% predopt.mode = "short_term";
predopt.mode = "long_term";
% predopt.mode = "interpolation";

% Added noise level
predopt.var_level = 0;
% predopt.var_level = 5;
% predopt.var_level = 10;

% Output display messages and figures is set to 1
predopt.out_disp = 1;
predopt.out_fig = 0;

% where is the data directory (folder that contains train, test data)
data_dir = strcat(filesep, "data", filesep); % platform agnostic filesep


%% Load data
switch predopt.mode
    case "short_term"
        problem_type = 1;

    case "long_term"
        problem_type = 2;

    case "interpolation"
        problem_type = 3;
end

% Create filename to load from load options
trainf_prefix_1 = "train_data_";
trainf_fname = trainf_prefix_1 + predopt.mode + "_" + ...
    num2str(predopt.var_level) + "_var.mat";

% full path and filename (train)
trainf_full = pwd() + data_dir + trainf_fname;

% full path and filename (test)
testf_prefix_1 = "test_data_";
testf_fname = testf_prefix_1 + predopt.mode + "_" + ...
    num2str(predopt.var_level) + "_var.mat";
testf_full = pwd() + data_dir + testf_fname;

% Load train and test data
load(trainf_full);
load(testf_full);

%% Data preprocessing
% sensor separation
time_o = train_data.time; % original time
time_o_ns = time_o(1:end-1); % original time except for last datetime
time_o_ys = time_o(2:end); % shifted by 1
time_shift = time_o_ns - time_o_ys; % time_shift should be mostly 3 seconds except for when sensor is changed
idx_sensor_change = find((time_shift > hours(36)) == 1); % find exactly where

% total number of sensors
num_sensors = size(idx_sensor_change,1) + 1;

% indices of sensors, start and end
idx_sensors = NaN(num_sensors,2);
idx_sensors(1,1) = 1;
idx_sensors(1:end-1,2) = idx_sensor_change;
idx_sensors(size(idx_sensors,1),2) = size(time_o,1);
idx_sensors(2:end,1) = idx_sensor_change + 1;

%% Preprocessing
% copy table for preprocessed data
train_data_ppc = train_data;
sensor_labels = strings(size(train_data,1),1); % labels for each sensor
sensor_counter_static = 0; % counter for static sensor
sensor_counter_mobile = 0; % counter for mobile sensor

% Begin preprocessing stage for each sensors
for i = 1:1:num_sensors
    % new table for each sensor
    sensi_tbl = train_data(idx_sensors(i,1):idx_sensors(i,2),:);

    % Create label for sensor
    var_in_lat = var(sensi_tbl.lat);
    if var_in_lat > 1e-10 % warning! very ad-hoc threshold-based solution!
        % it is mobile!
        sensor_counter_mobile = sensor_counter_mobile + 1;
        label = strcat("m", num2str(sensor_counter_mobile));

        idx_outlier = isoutlier(sensi_tbl.pm2d5,...
            "movmedian", minutes(30), ...
            "ThresholdFactor", 6, ...
            "SamplePoints", sensi_tbl.time);
    else
        % it is static!
        sensor_counter_static = sensor_counter_static + 1;
        label = strcat("s", num2str(sensor_counter_static));

        idx_outlier = isoutlier(sensi_tbl.pm2d5,...
            "median", ...
            "ThresholdFactor", 6, ...
            "SamplePoints", sensi_tbl.time);
    end
    
    sensor_labels(idx_sensors(i,1):idx_sensors(i,2),:) = label;
    

    % Output how many outliers there are in data
    if predopt.out_disp == 1
        disp("Number of outliers: " + num2str(sum(idx_outlier)));
    end
    
    % Visualize outliers
    if predopt.out_fig == 1
        % figure;
        % plot(sensi_tbl.time, sensi_tbl.pm2d5, 'ok');
        % hold on;
        % plot(sensi_tbl.time(idx_outlier), sensi_tbl.pm2d5(idx_outlier), 'xr');
        % hold off;
    end
    
    % Outlier removal via NaN fill
    % New table for individual sensor, to be filled with preprocessed PM2D5
    sensi_tbl_ppc = sensi_tbl;
    sensi_tbl_ppc.pm2d5(idx_outlier) = NaN;
    
    % Savitzky-Golay filter
    pm2d5_o = sensi_tbl_ppc.pm2d5;
    pm2d5_f0 = smoothdata(pm2d5_o, "sgolay", seconds(3*101), ...
        "Degree", 2, ...
        "SamplePoints", sensi_tbl_ppc.time);
    
    % Compare before and after S-G filter
    if predopt.out_fig == 1
        % figure;
        % plot(sensi_tbl_ppc.time, pm2d5_o, 'ko');
        % hold on;
        % plot(sensi_tbl_ppc.time, pm2d5_f0, 'rx');
        % hold off;
        % legend("original", "smoothdata with sgolay");
    end
    
    % Fill 'train_data_ppc.pm2d5' with preprocessed data
    train_data_ppc.pm2d5(idx_sensors(i,1):idx_sensors(i,2),:) = pm2d5_f0;    
end


%% Additional feature generation
% add sensor_labels to the preprocessed data
train_data_ppc.sensor_labels = sensor_labels;

% remove NaN from table
tbl_all = rmmissing(train_data_ppc);
% tbl_all = train_data_ppc;

% Cyclic features related to-
% for weekday (1~7)
[day_sin, day_cos] = cyc_feat_transf(weekday(tbl_all.time),7);
tbl_all.day_sin = day_sin;
tbl_all.day_cos = day_cos;

% for hour of day (0~23)
[hour_sin, hour_cos] = cyc_feat_transf(hour(tbl_all.time),24);
tbl_all.hour_sin = hour_sin;
tbl_all.hour_cos = hour_cos;

% for minute of time (0~59) - only needed for interpolation- deprecated?
% [min_sin, min_cos] = cyc_feat_transf(minute(tbl_all.time),60);
% tbl_all.min_sin = min_sin;
% tbl_all.min_cos = min_cos;


%% GPR
switch predopt.mode % which features to use depending on problem
    case "long_term"
        subset_table = {'pm2d5', 'hmd', 'tmp', 'lat', 'lon', ...
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos'};

    case "short_term"
        subset_table = {'pm2d5', 'hmd', 'tmp', 'lat', 'lon', ...
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos'};

    case "interpolation"
        subset_table = {'pm2d5', 'hmd', 'tmp', 'lat', 'lon', ...
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos'};
end

switch predopt.stage
    case "training_local"
        num_smpl = sum(predopt.train_local_sizes);
        switch predopt.sampling
            % Random sampling from the entire training data
            case "random"
                idx_samp = randsample( ...
                    transpose(1:1:size(tbl_all,1)),num_smpl);

                sens_group = categorical(tbl_all.sensor_labels(idx_samp));
                tbl_subset = tbl_all(idx_samp, subset_table);
                
            % Temporal order, e.g. the last parts of the data
            case "last_num_smpl"
                test_size = predopt.train_local_sizes(1,2);
                tbl_all_sorted = sortrows(tbl_all,"time");
                idx_train_time_start = size(tbl_all_sorted,1) - num_smpl + 1;
                idx_train_time_end = size(tbl_all_sorted,1);
                idx_samp = transpose(idx_train_time_start:1:idx_train_time_end);
                
                sens_group = categorical(tbl_all_sorted.sensor_labels(idx_samp));
                tbl_subset = tbl_all_sorted(idx_samp, subset_table);

            case "hybrid" % only for test!!! for now.

                tbl_all_sorted = sortrows(tbl_all,"time");
                
                % test size
                test_size = predopt.train_local_sizes(1,2);
                test_idx_end = size(tbl_all_sorted,1);
                test_idx_start = size(tbl_all_sorted,1) - test_size + 1;
                test_idxs = transpose(test_idx_start:1:test_idx_end);
                                
                % training from past+random
                train_size = predopt.train_local_sizes(1,1);
                train_size_past = floor(train_size*predopt.sampling_hybrid_past_ratio);
                train_size_rand = train_size - train_size_past;

                % training from past
                train_past_idx_end = test_idx_start - 1;
                train_past_idx_start = test_idx_start - 1 - train_size_past + 1;
                train_past_idxs = transpose(train_past_idx_start:1:train_past_idx_end);

                % training from random
                train_rand_idxs = randsample( ...
                    transpose(1:1:train_past_idx_start-1),train_size_rand);
                
                % the entire train+test dataset idxs
                idx_samp = vertcat(train_rand_idxs, train_past_idxs, test_idxs);

                sens_group = categorical(tbl_all_sorted.sensor_labels(idx_samp));
                tbl_subset = tbl_all_sorted(idx_samp, subset_table);

                % idx_train_time_start = size(tbl_all,1) - floor(num_smpl/2) + 1;
                % idx_train_time_end = size(tbl_all,1);
                % idx_samp_last = transpose(idx_train_time_start:1:idx_train_time_end);
                % idx_samp_rand = randsample( ...
                %     transpose(1:1:size(tbl_all,1)),num_smpl - floor(num_smpl/2));
                % idx_samp = vertcat(idx_samp_last, idx_samp_rand);
                
        end

    case "training"
        tbl_subset = tbl_all(:, subset_table);
        sens_group = categorical(sensor_labels);

    case "test"
        tbl_subset = tbl_all(:, subset_table);
        sens_group = categorical(sensor_labels);
end

% for training/hyperparamopt

% prepare categorical array for stratified CV

switch predopt.sampling

    case "random"
        cvp = cvpartition(sens_group, "Holdout", 0.2, "Stratify",true);
        

    case "last_num_smpl"
        test_idxs_bool = false(size(tbl_subset,1),1);
        test_idxs_bool(end-test_size+1:end) = true;
        cvp = cvpartition("CustomPartition", test_idxs_bool);

    case "hybrid"
        test_idxs_bool = false(size(tbl_subset,1),1);
        test_idxs_bool(end-test_size+1:end) = true;
        cvp = cvpartition("CustomPartition", test_idxs_bool);

end


% train GPR, hyperparameter optimization
y_true = tbl_subset.pm2d5(cvp.test,:);

% glmMdl = fitglm(tbl_subset, 'pm2d5~hmd+tmp+lat+lon');
% y_pred_glm = predict(glmMdl,tbl_subset(cvp.test,:));
% fit_nrmse_glm = goodnessOfFit(y_pred_glm,y_true,'NRMSE');

% gprMdl = fitrgp(tbl_subset, 'pm2d5',...
%         'FitMethod', 'fic', 'PredictMethod', 'fic', 'Standardize', 1,...
%         'BasisFunction','constant',...
%         'OptimizeHyperparameters', ...
%         'all',...
%         'Optimizer','fmincon',...
%         'HyperparameterOptimizationOptions', ...
%         struct('MaxObjectiveEvaluations', 50, ...
%         'UseParallel', false, ...
%         'SaveIntermediateResults', true, ...
%         'MaxTime', 60*60*8, ... % 8 hours running time limit
%         'Verbose', predopt.out_disp, ...
%         'CVPartition', cvp));
% y_pred = predict(gprMdl,tbl_subset(cvp.test,:));

% Using best hyperparameter so far...
fit_pred_method = 'exact';
% fit_pred_method = 'fic';

gprMdl = fitrgp(tbl_subset, 'pm2d5', ...
        'BasisFunction', 'linear',...
        'KernelFunction', 'ardmatern32',...
        'FitMethod',fit_pred_method,'PredictMethod',fit_pred_method, ...
        'CrossVal','on', 'CVPartition',cvp);

y_pred_all = kfoldPredict(gprMdl);
y_pred = y_pred_all(cvp.test,:);

% new_tmp_mean = repmat(mean(tbl_subset.tmp),size(tbl_subset,1),1);
% tbl_subset_notmp = tbl_subset;

% tbl_subset_notmp.tmp = new_tmp_mean;

if predopt.out_fig == 1
    figure;
    plot(y_true);
    hold on;
    plot(y_pred);
end

% y_pred_notmp = predict(gprMdl,tbl_subset_notmp(cvp.test,:));

fit_nrmse = goodnessOfFit(y_pred,y_true,'NRMSE');

if predopt.out_disp == 1
        disp("NRMSE: " + num2str(fit_nrmse));
end

res_save_name = strcat("GPR_", ...
    predopt.stage, "_", ...
    predopt.mode, "_var_", ...
    num2str(predopt.var_level,"%02d"), "_", ...
    string(datetime('now'),'yyyy-MM-dd_HHmmss'), ".mat");

save(res_save_name);


% for test dataset prediction


% ypred = kfoldPredict(cvgprMdl_trial);

if predopt.out_fig == 1 % not modified for hybrid sampling yet!
    % figure;
    % time_test = tbl_all.time(cvgprMdl_trial.Partition.test);
    % scatter(time_test, ypred(cvgprMdl_trial.Partition.test));
    % hold on;
    % y = table2array(tbl_subset(:,1));
    % scatter(time_te, y(cvgprMdl_trial.Partition.test),'r.');
    % % axis([0 1050 0 30]);
    % xlabel('Time');
    % ylabel('pmd2d5');
    % hold off;
end




%% Basic time series plots
% C_1 = linspecer(6);
% static_slc = 1:1:6;
% mobile_slc = 7:1:13;
% lgd_ss = strings(size(static_slc,2));
% lgd_ms = strings(size(mobile_slc,2)); 
% 
% fig_ts_ss = figure;
% tl_ss = tiledlayout(size(static_slc,2),1);
% ax_ss = gobjects(1,size(static_slc,2));
% for idx_ss = static_slc
%     ax_ss_i = nexttile;
%     ax_ss(1,idx_ss) = ax_ss_i;
%     nan_idx = isnan(comb_table.pm2d5(:,idx_ss));
%     plot(comb_table.time(~nan_idx), comb_table.pm2d5(~nan_idx,idx_ss),...
%         'Marker', '.', 'LineStyle', '-', 'Color', C_1(idx_ss,:));
%     title(strcat("Static Sensor ", num2str(idx_ss)));
% end
% 
% linkaxes(ax_ss,'xy');
% legend(num2str(static_slc'));



%% Plotting

%% Geoplot
% plot_markers = [repmat("x", 1, size(tjlt.data_static,2)),...
%     "o", "square", "diamond", "^", "v", ">", "<"];
% 
% geofig_1 = figure('Visible','off');
% % geofig_1 = figure('Visible','on');
% 
% Frames_dt = days(1);
% numFrames = days(dt_end - dt_start);
% frameIndex = 0;
% frames = struct('cdata',[],'colormap',[]);
% frames(numFrames) = frames;
% 
% dt_t0 = dt_start - Frames_dt;
% dt_t1 = dt_start;
% 
% for day_idx = 1:1:numFrames
%     % Prep frame
%     frameIndex = frameIndex + 1;
%     % Prep data
%     dt_t0 = dt_t0 + Frames_dt;
%     dt_t1 = dt_t1 + Frames_dt;
%     part_dt_idx = comb_table.time >= dt_t0 & comb_table.time < dt_t1;
%     comb_table_i = comb_table(part_dt_idx,:);
%     for sens_idx = 1:1:size(comb_table_i.pm2d5,2)
%         lat_i = comb_table_i.lat(:,sens_idx);
%         lon_i = comb_table_i.lon(:,sens_idx);
%         pm2d5_i = comb_table_i.pm2d5(:,sens_idx);
%         % if sens_idx <= 6
%         % geoscatter(lat_i, lon_i, 50, pm2d5_i, "filled", "Marker", plot_markers(1,sens_idx));
%         geoscatter(lat_i, lon_i, 50, pm2d5_i, "Marker", plot_markers(1,sens_idx));
%         hold on;
%         % elseif sens_idx >= 7
%             % geoscatter(lat_i, lon_i, 50, pm2d5_i, "Marker",plot_markers(1,sens_idx));
%         % end
%     end
%     hold off;
%     geolimits([39.00, 39.25],[117.65, 117.85]);
%     % geolimits([39.095, 39.13],[117.725, 117.755]);
% 
%     geobasemap 'streets-light';
%     plot_title = strcat("From ", string(dt_t0), " to ", string(dt_t1));
%     title(plot_title);
%     colorbar;
%     clim([0, 150]);
% 
%     % Get frames
%     frames(frameIndex) = getframe(geofig_1);
% 
% end
% 
% animated(1,1,1,numFrames) = 0;
% for k=1:numFrames
%    if k == 1
%       [animated,cmap] = rgb2ind(frames(k).cdata,256,'nodither');
%    else
%       animated(:,:,1,k) = ...
%          rgb2ind(frames(k).cdata,cmap,'nodither');
%    end     
% end
% 
% filename = 'pm2d5_animated_all.gif';
% imwrite(animated,cmap,filename,'DelayTime',0.5, ...
%    'LoopCount',inf);
% web(filename);

%% In-line functions
% function [tbl_all] = vertcat_tables(ld_file)
%     % new giant table, with info about where it came from
%     tbl_all = table();
% 
%     for idx_ss = 1:1:size(ld_file.data_static,2)
%         some_table = ld_file.data_static{1,idx_ss};
%         some_table.sensor_type = repmat("static",size(some_table,1),1);
%         some_table.sensor_num = repmat(idx_ss,size(some_table,1),1);
%         tbl_all = vertcat(tbl_all, some_table);
%     end
% 
%     for idx_ms = 1:1:size(ld_file.data_mobile,2)
%         some_table = ld_file.data_mobile{1,idx_ms};
%         some_table.sensor_type = repmat("mobile",size(some_table,1),1);
%         some_table.sensor_num = repmat(idx_ms,size(some_table,1),1);
%         tbl_all = vertcat(tbl_all, some_table);
%     end
% end

% function [feat_sin, feat_cos] = cyc_feat_transf(data, period)
%     feat_sin = sin(2*pi*data/period);
%     feat_cos = cos(2*pi*data/period);
% 
% end


%% Deprecated
% function [comb_table] = comb_tables(lddt,time)
%     % Input params:
%     % lddt: loaded data
% 
%     % Create NaN for humidity, speed, temp., pm2d5, lat, lon
%     dat_ncol = size(lddt.data_static,2) + size(lddt.data_mobile,2);
%     dat_nrow = size(time,1);
% 
%     humidity = NaN(dat_nrow,dat_ncol);
%     speed = NaN(dat_nrow,dat_ncol);
%     temperature = NaN(dat_nrow,dat_ncol);
%     pm2d5 = NaN(dat_nrow,dat_ncol);
%     lat = NaN(dat_nrow,dat_ncol);
%     lon = NaN(dat_nrow,dat_ncol);
% 
%     comb_table = table(time,humidity,speed,temperature,pm2d5,lat,lon);
% 
%     for idx_s = 1:1:size(lddt.data_static,2)
%         [LocA, LocB] = ismember(lddt.data_static{1,idx_s}.time,time);
%         comb_table.humidity(LocB,idx_s) = lddt.data_static{1,idx_s}.humidity(LocA, :);
%         comb_table.speed(LocB,idx_s) = lddt.data_static{1,idx_s}.speed(LocA, :);
%         comb_table.temperature(LocB,idx_s) = lddt.data_static{1,idx_s}.temperature(LocA, :);
%         comb_table.pm2d5(LocB,idx_s) = lddt.data_static{1,idx_s}.pm2d5(LocA, :);
%         comb_table.lat(LocB,idx_s) = lddt.data_static{1,idx_s}.lat(LocA, :);
%         comb_table.lon(LocB,idx_s) = lddt.data_static{1,idx_s}.lon(LocA, :);
%     end
% 
%     % start index from end of num. of static sensors
%     idx_m_start = size(lddt.data_static,2);
%     for idx_m = 1:1:size(lddt.data_mobile,2)
%         [LocA, LocB] = ismember(lddt.data_mobile{1,idx_s}.time,time);
%         idx_mt = idx_m_start + idx_m;
%         comb_table.humidity(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.humidity(LocA, :);
%         comb_table.speed(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.speed(LocA, :);
%         comb_table.temperature(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.temperature(LocA, :);
%         comb_table.pm2d5(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.pm2d5(LocA, :);
%         comb_table.lat(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.lat(LocA, :);
%         comb_table.lon(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.lon(LocA, :);
%     end
% 
% end

