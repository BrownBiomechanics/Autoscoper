function [outputArg1,outputArg2] = saveTracking(autoscoper_socket, volume, filename)
%SAVETRACKING Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[3 typecast(int32(volume),'uint8') filename]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

