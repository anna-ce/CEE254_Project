%% Script for Prediction of PM2D5
clear; clc; close all;

%% Analysis Control

% Hyp. param. opt. (training) or making predictions for test data (test)
predopt.stage = "training";
% predopt.stage = "test";

% Which type of prediction to make
predopt.mode = "short_term";
% predopt.mode = "long_term";
% predopt.mode = "interpolation";

% Added noise level
predopt.var_level = 0;
% predopt.var_level = 5;
% predopt.var_level = 10;

% Output display messages and figures is set to 1
predopt.out_disp = 0;
predopt.out_fig = 0;

%% Load data
trainf_prefix_1 = "train_data_";

switch predopt.mode
    case "short_term"
        problem_type = 1;

    case "long_term"
        problem_type = 2;

    case "interpolation"
        problem_type = 3;
end

trainf_fname = trainf_prefix_1 + predopt.mode + "_" + ...
    num2str(predopt.var_level) + "_var.mat";

data_dir = strcat(filesep, "data", filesep); % platform agnostic filesep

trainf_full = pwd() + data_dir + trainf_fname;

testf_prefix_1 = "test_data_";
testf_fname = testf_prefix_1 + predopt.mode + "_" + ...
    num2str(predopt.var_level) + "_var.mat";
testf_full = pwd() + data_dir + testf_fname;

% Load train and test data
load(trainf_full);
load(testf_full);

%% Data preprocessing
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
train_data_ppc = train_data;
sensor_labels = strings(size(train_data,1),1);
sensor_counter_static = 0;
sensor_counter_mobile = 0;

% Begin preprocessing stage for all sensors
for i = 1:1:num_sensors
    % new table for each sensor
    sensi_tbl = train_data(idx_sensors(i,1):idx_sensors(i,2),:);
    
    % Outlier removal
    % TODO tune this so that it does not get rid of legitimate data points
    idx_outlier = isoutlier(sensi_tbl.pm2d5,...
        "movmedian", minutes(30), ...
        "ThresholdFactor", 6, ...
        "SamplePoints", sensi_tbl.time);

    % Output how many outliers there are in data
    if predopt.out_disp == 1
        disp("Number of outliers: " + num2str(sum(idx_outlier)));
    end

    % Visualize outliers
    if predopt.out_fig == 1
        figure;
        plot(sensi_tbl.time, sensi_tbl.pm2d5, 'ok');
        hold on;
        plot(sensi_tbl.time(idx_outlier), sensi_tbl.pm2d5(idx_outlier), 'xr');
        hold off;
    end

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
        figure;
        plot(sensi_tbl_ppc.time, pm2d5_o, 'ko');
        hold on;
        plot(sensi_tbl_ppc.time, pm2d5_f0, 'rx');
        hold off;
        legend("original", "smoothdata with sgolay");
    end
    
    % Fill 'train_data_ppc.pm2d5' with preprocessed data
    train_data_ppc.pm2d5(idx_sensors(i,1):idx_sensors(i,2),:) = pm2d5_f0;

    % Create label for sensor
    var_in_lat = var(sensi_tbl_ppc.lat);
    if var_in_lat > 1e-10 % very ad-hoc solution! watch out!
        sensor_counter_mobile = sensor_counter_mobile + 1;
        label = strcat("m", num2str(sensor_counter_mobile));
    else
        sensor_counter_static = sensor_counter_static + 1;
        label = strcat("s", num2str(sensor_counter_static));
    end
    sensor_labels(idx_sensors(i,1):idx_sensors(i,2),:) = label;
end

%% Additional feature generation
tbl_all = train_data_ppc;
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
switch predopt.mode
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

tbl_subset = tbl_all(:, subset_table);

% cvgprMdl_trial = fitrgp(tbl_subset, 'pm2d5', ...
%     'KernelFunction','ardsquaredexponential',...
%     'FitMethod', 'fic', 'PredictMethod', 'fic',...
%     'Standardize',true, 'Holdout', 0.3);

% for training/hyperparamopt

% prepare categorical array for stratified CV
sens_group = categorical(sensor_labels);
cvp = cvpartition(sens_group, "Holdout", 0.2, "Stratify",true);

% train GPR, hyperparameter optimization
gprMdl = fitrgp(tbl_subset, 'pm2d5',...
        'FitMethod', 'fic', 'PredictMethod', 'fic', 'Standardize', 1,...
        'BasisFunction','constant',...
        'OptimizeHyperparameters', {'KernelFunction','KernelScale','Sigma'},...
        'Optimizer','fmincon',...
        'HyperparameterOptimizationOptions', ...
        struct('MaxObjectiveEvaluations', 50, ...
        'UseParallel', false, ...
        'CVPartition', cvp));

res_save_name = strcat("GPR_", ...
    predopt.stage, "_", ...
    predopt.mode, "_var_", ...
    num2str(predopt.var_level,"%02d"), "_", ...
    string(datetime('now'),'yyyy-MM-dd_Hmmss'), ".mat");

save(res_save_name);

% gprMdl = fitrgp(tbl_subset, 'pm2d5','KernelFunction','ardrationalquadratic',...
%         'FitMethod','fic','PredictMethod','fic','Standardize',1,...
%         'OptimizeHyperparameters',{'Sigma'},...
%         'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',50),...
%         'Optimizer','fmincon');

% for test dataset prediction


% ypred = kfoldPredict(cvgprMdl_trial);

if predopt.out_fig == 1
    figure;
    time_test = tbl_all.time(cvgprMdl_trial.Partition.test);
    scatter(time_test, ypred(cvgprMdl_trial.Partition.test));
    hold on;
    y = table2array(tbl_subset(:,1));
    scatter(time_te, y(cvgprMdl_trial.Partition.test),'r.');
    % axis([0 1050 0 30]);
    xlabel('Time');
    ylabel('pmd2d5');
    hold off;
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

function [feat_sin, feat_cos] = cyc_feat_transf(data, period)
    feat_sin = sin(2*pi*data/period);
    feat_cos = cos(2*pi*data/period);

end


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

