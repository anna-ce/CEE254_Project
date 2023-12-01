clear; clc;
load train_data_long_term_0_var

time = train_data.time;
uniqueDays = unique(dateshift(time, 'start', 'day'));

train_data_o = train_data;

for j = 1:length(uniqueDays)

    day = uniqueDays(j);
    indices = find(dateshift(time, 'start', 'day') == day);

    for myhour = 0:23

        hourIndices = find(hour(time(indices, 1)) == myhour);

        if ~isempty(hourIndices)

            pm2d5_hour = train_data.pm2d5(indices(hourIndices));
            outliers = isoutlier(pm2d5_hour,"median", );
            % disp(sum(outliers));
            train_data.pm2d5(indices(hourIndices(outliers))) = NaN;

        end
    end
end



    %only filling for NaN values, not for missing time steps

    % padd
    time_column = train_data.time;
    mean_values = mean(train_data{:, ~strcmp(train_data.Properties.VariableNames, 'time')}, 'omitnan');
    mean_start = array2table(mean_values, 'VariableNames', train_data.Properties.VariableNames(~strcmp(train_data.Properties.VariableNames, 'time')));
    mean_end = array2table(mean_values, 'VariableNames', train_data.Properties.VariableNames(~strcmp(train_data.Properties.VariableNames, 'time')));
    mean_start.time = train_data.time(1);
    mean_end.time = train_data.time(end);
    train_data = [mean_start; train_data; mean_end];


    % Fill missing pm2d5 values with the average of the value before and after
    train_data = fillmissing(train_data, 'linear', 'DataVariables', 'pm2d5');
    train_data(end, :) = [];
    train_data(1, :) = [];
    train_data = movevars(train_data, 'time', 'Before','hmd');


    Fs = 1/(3);

    % Apply a moving average filter
    % windowSize = 20*30; %30 minute moving average: (3(s)*20){=1 minute}*30 = 30 minutes
    windowSize = 20*5; %30 minute moving average: (3(s)*20){=1 minute}*30 = 30 minutes
    pm2d5_s = movmean(train_data.pm2d5, windowSize);
    hmd_s = movmean(train_data.hmd, windowSize);
    tmp_s = movmean(train_data.tmp, windowSize);    
    % pm2d5_s = train_data.pm2d5;
    % tmp_s = train_data.tmp;
    % hmd_s = train_data.hmd;

    % Butter Filter
    % fc = 1/(30*60); %60(s){=1 minute}*30 = 30 minutes
    fc = 1/(5*60); %60(s){=1 minute}*30 = 30 minutes
    [b, a] = butter(4,fc./(Fs./2));
    pm2d5_f1 = filter(b,a,pm2d5_s);
    hmd_f1 = filter(b,a,hmd_s);
    tmp_f1 = filter(b,a,tmp_s);

    % Calculate the signal-to-noise ratio (SNR)
    signal_power = rms(pm2d5_f1)^2; % Power of the filtered signal
    noise_power = rms(train_data.pm2d5 - pm2d5_f1)^2; % Power of the noise
    snr_db = 10 * log10(signal_power / noise_power); % SNR in decibels

    %Store clean data in new table and then transfer it to new cell array
    train_data_clean = table(train_data.time,hmd_f1,train_data.spd,tmp_f1,pm2d5_f1,train_data.lat,train_data.lon);
    train_data_clean = renamevars(train_data_clean,["Var1" "Var2" "Var3" "Var4" "Var5" "Var6" "Var7"],["time" "hmd" "spd" "tmp" "pm2d5" "lat" "lon" ]);
    figure;
    % plot(train_data.time(1:100000), pm2d5_s(1:100000))
    plot(train_data_clean.time(1:100000),train_data_clean.pm2d5(1:100000),'.-r')
    hold on;
    plot(train_data.time(1:100000),train_data.pm2d5(1:100000),'.-k')





