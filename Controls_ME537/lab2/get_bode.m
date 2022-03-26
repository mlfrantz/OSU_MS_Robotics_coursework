function [freq_voltage, freq_table, freq_motor, gain_table, phase_angle_table, gain_motor, phase_angle_motor] = get_bode(pos_motor, pos_table, voltage)
%GET_BODE Summary of this function goes here
%   Detailed explanation goes here

Fs = 1000;     % Sampling Frequency (1msec)
T = 1/Fs;       % Sampling Period
L = 10000;

Y_voltage = fft(voltage);
P2 = abs(Y_voltage/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

[max_val_voltage, ind] = max(P1(2:end)); % Highest non zero frequency
freq_voltage = f(ind); % Actual output frequency

Y_table = fft(pos_table);
P2 = abs(Y_table/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

[max_val_table, ind] = max(P1(10:end)); % Highest non zero frequency
freq_table = f(ind); % Actual output frequency

Y_motor = fft(pos_motor);
P2 = abs(Y_motor/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

[max_val_motor, ind] = max(P1(10:end)); % Highest non zero frequency
freq_motor = f(ind); % Actual output frequency

H_table = Y_table/Y_voltage;
gain_table = 20*log10(abs(H_table)); % in dB
phase_angle_table = angle(H_table)*180/pi; % In Degrees

H_motor = Y_motor/Y_voltage;
gain_motor = 20*log10(abs(H_motor)); % in dB
phase_angle_motor = angle(H_motor)*180/pi; % In Degrees

% Outputs: Gain, Phase, Frequency, 

end

