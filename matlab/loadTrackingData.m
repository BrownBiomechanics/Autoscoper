function loadTrackingData(autoscoper_socket, volume, trackingData)
%LOADTRACKINGDATA Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[2 typecast(int32(volume),'uint8') trackingData]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end

