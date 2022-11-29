function [outputArg1,outputArg2] = saveTracking(autoscoper_socket, volume, filename,save_as_matrix,save_as_rows,save_with_commas,convert_to_cm,convert_to_rad,interpolate)
%SAVETRACKING Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[3 typecast(int32(volume),'uint8') typecast(int32(save_as_matrix),'uint8') typecast(int32(save_as_rows),'uint8') typecast(int32(save_with_commas),'uint8') typecast(int32(convert_to_cm),'uint8') typecast(int32(convert_to_rad),'uint8') typecast(int32(interpolate),'uint8') filename]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

