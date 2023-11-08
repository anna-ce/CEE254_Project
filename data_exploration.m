%% Script for Data Exploration
clear; clc; close all;

tjlt = load("long_term_tianjin_train_val.mat");
tjst = load("short_term_foshan_train_val.mat");
% load("long_term_tianjin_train_val.mat");

dt_start = datetime("2018-04-24T00:00",'InputFormat','uuuu-MM-dd''T''HH:mm');
dt_end = datetime("2018-05-22T00:00",'InputFormat','uuuu-MM-dd''T''HH:mm');

time = transpose(dt_start:minutes(1):dt_end);

comb_table = comb_tables(tjlt,time);

% Weekly
% how many weeks in data? 
% find indices for each week
disp(strcat("There are ", num2str(days(dt_end - dt_start)/7), " week(s) in data"));
num_weeks = floor(days(dt_end - dt_start)/7);


%% Encoding cyclical datetime info. as features
% for weekday (1~7)
[day_sin, day_cos] = cyc_feat_transf(weekday(comb_table.time),7);
comb_table.day_sin = day_sin;
comb_table.day_cos = day_cos;

% for hour of day (0~23)
[hour_sin, hour_cos] = cyc_feat_transf(hour(comb_table.time),24);
comb_table.hour_sin = hour_sin;
comb_table.hour_cos = hour_cos;

figure;
% scatter(comb_table.day_sin, comb_table.pm2d5);
scatter(comb_table.day_cos, comb_table.day_sin);

%% Basic time series plots
C_1 = linspecer(6);
static_slc = 1:1:6;
mobile_slc = 7:1:13;
lgd_ss = strings(size(static_slc,2));
lgd_ms = strings(size(mobile_slc,2)); 

fig_ts_ss = figure;
tl_ss = tiledlayout(size(static_slc,2),1);
ax_ss = gobjects(1,size(static_slc,2));
for idx_ss = static_slc
    ax_ss_i = nexttile;
    ax_ss(1,idx_ss) = ax_ss_i;
    nan_idx = isnan(comb_table.pm2d5(:,idx_ss));
    plot(comb_table.time(~nan_idx), comb_table.pm2d5(~nan_idx,idx_ss),...
        'Marker', '.', 'LineStyle', '-', 'Color', C_1(idx_ss,:));
    title(strcat("Static Sensor ", num2str(idx_ss)));
end

linkaxes(ax_ss,'xy');
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
function [comb_table] = comb_tables(lddt,time)
    % Input params:
    % lddt: loaded data
    
    % Create NaN for humidity, speed, temp., pm2d5, lat, lon
    dat_ncol = size(lddt.data_static,2) + size(lddt.data_mobile,2);
    dat_nrow = size(time,1);

    humidity = NaN(dat_nrow,dat_ncol);
    speed = NaN(dat_nrow,dat_ncol);
    temperature = NaN(dat_nrow,dat_ncol);
    pm2d5 = NaN(dat_nrow,dat_ncol);
    lat = NaN(dat_nrow,dat_ncol);
    lon = NaN(dat_nrow,dat_ncol);

    comb_table = table(time,humidity,speed,temperature,pm2d5,lat,lon);

    for idx_s = 1:1:size(lddt.data_static,2)
        [LocA, LocB] = ismember(lddt.data_static{1,idx_s}.time,time);
        comb_table.humidity(LocB,idx_s) = lddt.data_static{1,idx_s}.humidity(LocA, :);
        comb_table.speed(LocB,idx_s) = lddt.data_static{1,idx_s}.speed(LocA, :);
        comb_table.temperature(LocB,idx_s) = lddt.data_static{1,idx_s}.temperature(LocA, :);
        comb_table.pm2d5(LocB,idx_s) = lddt.data_static{1,idx_s}.pm2d5(LocA, :);
        comb_table.lat(LocB,idx_s) = lddt.data_static{1,idx_s}.lat(LocA, :);
        comb_table.lon(LocB,idx_s) = lddt.data_static{1,idx_s}.lon(LocA, :);
    end
    
    % start index from end of num. of static sensors
    idx_m_start = size(lddt.data_static,2);
    for idx_m = 1:1:size(lddt.data_mobile,2)
        [LocA, LocB] = ismember(lddt.data_mobile{1,idx_s}.time,time);
        idx_mt = idx_m_start + idx_m;
        comb_table.humidity(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.humidity(LocA, :);
        comb_table.speed(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.speed(LocA, :);
        comb_table.temperature(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.temperature(LocA, :);
        comb_table.pm2d5(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.pm2d5(LocA, :);
        comb_table.lat(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.lat(LocA, :);
        comb_table.lon(LocB,idx_mt) = lddt.data_mobile{1,idx_s}.lon(LocA, :);
    end

end

function [feat_sin, feat_cos] = cyc_feat_transf(data, period)
    feat_sin = sin(2*pi*data/period);
    feat_cos = cos(2*pi*data/period);

end