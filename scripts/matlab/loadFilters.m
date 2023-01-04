function loadFilters(autoscoper_socket,cameraId, filtersConfig)
%LOADFILTERS Summary of this function goes here
%   Detailed explanation goes here
fwrite(autoscoper_socket,[4 typecast(int32(cameraId),'uint8') filtersConfig]);
while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
end
