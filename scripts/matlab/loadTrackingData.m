function loadTrackingData(autoscoper_socket, volume, trackingData,save_as_matrix,save_as_rows,save_with_commas,convert_to_cm,convert_to_rad,interpolate)
%LOADTRACKINGDATA Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[2 typecast(int32(volume),'uint8') typecast(int32(save_as_matrix),'uint8') typecast(int32(save_as_rows),'uint8') typecast(int32(save_with_commas),'uint8') typecast(int32(convert_to_cm),'uint8') typecast(int32(convert_to_rad),'uint8') typecast(int32(interpolate),'uint8') trackingData]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

