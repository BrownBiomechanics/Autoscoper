function loadTrial(autoscoper_socket, trial_file)
%LOAD_TRIAL Summary of this function goes here
%   Detailed explanation goes here
%Load trial
fwrite(autoscoper_socket,[1 trial_file]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

