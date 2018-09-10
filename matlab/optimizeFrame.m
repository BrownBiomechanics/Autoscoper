function optimizeFrame(autoscoper_socket,volumeID, frame,repeats,max_iter,min_lim,max_lim,max_stall_iter)
%optimizeFrame Summary of this function goes here
%   Detailed explanation goes here

fwrite(autoscoper_socket,...
    [11 typecast(int32(volumeID),'uint8') typecast(int32(frame),'uint8') typecast(int32(repeats),'uint8') ...
    typecast(int32(max_iter),'uint8') typecast(double(min_lim),'uint8') ...
    typecast(double(max_lim),'uint8') typecast(int32(max_stall_iter),'uint8')]);

while autoscoper_socket.BytesAvailable == 0
    pause(1)
end
data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);

end

